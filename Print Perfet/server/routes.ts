import type { Express } from "express";
import type { Server } from "http";
import { storage } from "./storage";
import { api } from "@shared/routes";
import { z } from "zod";
import OpenAI from "openai";
import { GoogleGenAI } from "@google/genai";
import Anthropic from "@anthropic-ai/sdk";
import { getLoopStatus } from "./learning-loop";

function getPerplexityClient() {
  const apiKey = process.env.PERPLEXITY_API_KEY;
  if (!apiKey) throw new Error("PERPLEXITY_API_KEY environment variable is not set");
  return new OpenAI({
    apiKey,
    baseURL: "https://api.perplexity.ai",
  });
}

function getGptClient() {
  const apiKey = process.env.AI_INTEGRATIONS_OPENAI_API_KEY;
  if (!apiKey) throw new Error("AI_INTEGRATIONS_OPENAI_API_KEY environment variable is not set");
  return new OpenAI({
    apiKey,
    baseURL: process.env.AI_INTEGRATIONS_OPENAI_BASE_URL,
  });
}

function getGeminiClient() {
  const apiKey = process.env.AI_INTEGRATIONS_GEMINI_API_KEY;
  if (!apiKey) throw new Error("AI_INTEGRATIONS_GEMINI_API_KEY environment variable is not set");
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      apiVersion: "",
      baseUrl: process.env.AI_INTEGRATIONS_GEMINI_BASE_URL,
    },
  });
}

function getClaudeClient() {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) throw new Error("ANTHROPIC_API_KEY environment variable is not set");
  return new Anthropic({ apiKey });
}

function lessonToText(lesson: {
  title?: string | null; problem?: string | null; rootCause?: string | null;
  fix?: string | null; prevention?: string | null; projectContext?: string | null;
  trade?: string | null; tags?: string[] | null; phase?: string | null;
  triggerEvent?: string | null; conditions?: string | null; processSteps?: string | null;
  reworkAvoided?: string | null; safetyIncidentsAvoided?: string | null;
}): string {
  return [
    lesson.title,
    lesson.phase ? `Phase: ${lesson.phase}` : null,
    lesson.triggerEvent ? `Trigger: ${lesson.triggerEvent}` : null,
    lesson.conditions ? `Conditions: ${lesson.conditions}` : null,
    lesson.problem,
    lesson.rootCause,
    lesson.fix,
    lesson.processSteps,
    lesson.prevention,
    lesson.projectContext,
    lesson.reworkAvoided,
    lesson.safetyIncidentsAvoided,
    lesson.trade,
    ...(lesson.tags || []),
  ].filter(Boolean).join(" ");
}

async function generateEmbedding(text: string): Promise<number[] | null> {
  try {
    const gpt = getGptClient();
    const response = await gpt.embeddings.create({
      model: "text-embedding-3-small",
      input: text.slice(0, 8000),
    });
    return response.data[0].embedding;
  } catch (err) {
    console.error("[embeddings] Failed to generate:", err);
    return null;
  }
}

async function extractPdfText(base64: string): Promise<string> {
  try {
    const buffer = Buffer.from(base64, "base64");
    // pdf-parse v2 exports a default async function: pdfParse(buffer) => { text, ... }
    const pdfParse = ((await import("pdf-parse")) as any).default ?? (await import("pdf-parse") as any);
    const result = await pdfParse(buffer);
    return (result.text as string)?.trim() || "";
  } catch (err) {
    console.error("[pdf] Failed to extract text:", err);
    return "";
  }
}

async function autoAnalyzeDocument(docId: number, projectId: number): Promise<void> {
  try {
    const [doc, project, openRfis] = await Promise.all([
      storage.getDocument(docId),
      storage.getProject(projectId),
      storage.getRfis(projectId),
    ]);
    if (!doc || !doc.fileData) return;

    const trade = doc.trade || "general";
    const codes = TRADE_CODES[trade] || TRADE_CODES.general;
    const location = project?.location || "Unknown location";
    const openRfiList = openRfis.filter(r => r.status === "open").slice(0, 20)
      .map(r => `- [${r.rfiNumber || r.id}] ${r.trade.toUpperCase()}: ${r.title}`).join("\n");

    const systemPrompt = `You are an expert construction code inspector analyzing documents for Dollar Tree Store #10746, 1029 E Grand Ave, Rothschild, WI 54476.
Trade scope: ${trade.toUpperCase()} | Applicable codes: ${codes}
Open RFIs to cross-reference: ${openRfiList || "None"}
Analyze the provided document. Structure your response:
1. **Document Summary** — what this document contains
2. **Key Information** — important dates, amounts, contacts, specs, or requirements
3. **Trade Scope / Action Items** — work or actions required
4. **Code Compliance Notes** — any compliance considerations
5. **Issues / Red Flags** — conflicts, missing info, or items needing follow-up`;

    const gpt = getGptClient();
    let analysisResult: string;

    if (doc.fileMime?.startsWith("image/")) {
      const response = await gpt.chat.completions.create({
        model: "gpt-4o",
        max_tokens: 4096,
        messages: [
          { role: "system", content: systemPrompt },
          {
            role: "user",
            content: [
              { type: "image_url", image_url: { url: `data:${doc.fileMime};base64,${doc.fileData}`, detail: "high" } },
              { type: "text", text: `Read and analyze this document: "${doc.name}"` }
            ]
          }
        ],
      });
      analysisResult = response.choices[0]?.message?.content || "No analysis generated.";
    } else if (doc.fileMime === "application/pdf") {
      const pdfText = await extractPdfText(doc.fileData);
      if (!pdfText) {
        console.log(`[auto-analyze] PDF #${docId} has no extractable text — skipping`);
        return;
      }
      const response = await gpt.chat.completions.create({
        model: "gpt-4o",
        max_tokens: 4096,
        messages: [
          { role: "system", content: systemPrompt },
          { role: "user", content: `Document: "${doc.name}"\n\nFULL PDF CONTENT:\n${pdfText.slice(0, 12000)}\n\nAnalyze this document completely.` }
        ],
      });
      analysisResult = response.choices[0]?.message?.content || "No analysis generated.";
    } else {
      return;
    }

    await storage.updateDocumentContent(docId, analysisResult);
    console.log(`[auto-analyze] Document #${docId} "${doc.name}" analyzed successfully`);
  } catch (err: any) {
    console.error(`[auto-analyze] Document #${docId} failed: ${err.message}`);
  }
}

async function embedLessonAsync(lesson: { id: number; title?: string | null; problem?: string | null; rootCause?: string | null; fix?: string | null; prevention?: string | null; projectContext?: string | null; trade?: string | null; tags?: string[] | null }) {
  const text = lessonToText(lesson);
  const embedding = await generateEmbedding(text);
  if (embedding) {
    await storage.updateLessonEmbedding(lesson.id, embedding);
    console.log(`[embeddings] Embedded lesson #${lesson.id}`);
  }
}

const TRADE_CODES: Record<string, string> = {
  hvac: `WISCONSIN/ROTHSCHILD HVAC CODES (2025-2026):
- 2021 International Mechanical Code (IMC) — adopted as part of Wisconsin commercial package effective Oct 1, 2025
- Wisconsin Administrative Code SPS 362 (Heating, Ventilating, Air Conditioning)
- ASHRAE 90.1 Energy Standard for Buildings — enforced via 2021 IECC commercial provisions
- SMACNA Duct Construction Standards — duct leakage and fabrication
- NFPA 90A (2021): Standard for the Installation of Air-Conditioning and Ventilating Systems
- NFPA 90B (2024): Standard for the Installation of Warm Air Heating and Air-Conditioning Systems (KEY — 2024 edition adopted)
- EPA Section 608: Refrigerant handling and recovery — certified technician required
- NEC 210.8(F) 2023: GFCI protection required for outdoor HVAC equipment — compliance required by September 1, 2026
- Foundation/footings: Minimum 48 inches below finished grade (Wisconsin frost depth)`,

  electrical: `WISCONSIN/ROTHSCHILD ELECTRICAL CODES (2025-2026):
- 2023 National Electrical Code (NEC / NFPA 70) — adopted as of January 1, 2025 (replaces 2017 NEC)
- Wisconsin Administrative Code SPS 316 (Electrical)
- NEC 210.8(F) 2023: GFCI protection for all outdoor HVAC unit receptacles — required by September 1, 2026
- NFPA 72 (2025 edition): National Fire Alarm and Signaling Code — KEY for new commercial fire alarm systems
- NEC 110.26: Minimum working clearances at electrical panels — 36" depth, 30" width, 6.5" height
- NEC 700: Emergency lighting and exit signs
- Wisconsin Energy Code: 2021 IECC Commercial — lighting controls, power monitoring, EN-series compliance forms
- ADA: All accessible receptacle and device heights per ANSI A117.1`,

  plumbing: `WISCONSIN/ROTHSCHILD PLUMBING CODES (2025-2026):
- Wisconsin Administrative Code Chapters SPS 381-387 (full plumbing code range — replaces single SPS 382 reference)
  • SPS 381: General plumbing provisions
  • SPS 382: Plumbing systems (fixtures, piping, drainage)
  • SPS 383: Private onsite wastewater treatment
  • SPS 384: Plumbing products and materials
  • SPS 385: Existing plumbing systems
  • SPS 386: Manufactured buildings
  • SPS 387: Plumbing permit and inspection requirements
- Atmospheric vacuum breakers required at all mop sinks and hose bibs
- Water heater T&P relief valve discharge to approved drain required
- Foundation/footings: Minimum 48 inches below finished grade (frost depth)`,

  fire: `WISCONSIN/ROTHSCHILD FIRE PROTECTION CODES (2025-2026):
- NFPA 13 (2022): Standard for the Installation of Sprinkler Systems
- NFPA 25 (2023): Standard for the Inspection, Testing, and Maintenance of Water-Based Fire Protection Systems
- NFPA 72 (2025 edition): National Fire Alarm and Signaling Code — KEY edition for new commercial work in Wisconsin
- NFPA 90B (2024): Air distribution systems — key for HVAC/fire interaction
- IFC (International Fire Code) 2021 — adopted as part of Wisconsin commercial package Oct 1, 2025
- Local AHJ (Authority Having Jurisdiction): Rothschild/Marathon County fire marshal approval required
- Fire sprinkler shutdown: Coordinate with fire marshal; formal notification required before any tie-in or demo`,

  general: `WISCONSIN/ROTHSCHILD GENERAL CONSTRUCTION CODES (2025-2026):
- 2021 International Building Code (IBC) — Wisconsin officially transitioned from 2015 to 2021 IBC on October 1, 2025
- 2021 International Energy Conservation Code (IECC) — adopted simultaneously with 2021 IBC on Oct 1, 2025
- 2021 International Mechanical Code (IMC) — part of the new commercial package
- Wisconsin Administrative Code SPS 361-366: Commercial building construction
- ADA Standards for Accessible Design (2010) + ANSI A117.1: All accessible routes, restrooms, doors, hardware
- OSHA 29 CFR 1926: Construction safety standards
- Foundation/footings: Minimum 48 inches below finished grade (Wisconsin frost depth requirement)
- IBC Table 1017.2: Maximum egress travel distance 250ft in sprinklered retail (200ft without sprinklers)
- IBC 2406: Safety glazing required within 24" of doors and within 18" of floor
- Wisconsin Energy Code: COMcheck compliance required — EN-series forms must be signed and submitted`,
};

const PUNCHLIST_SEED: Omit<Parameters<typeof storage.createPunchlistItem>[0], 'projectId'>[] = [
  // ── STOCK ROOM ─────────────────────────────────────────────────────────────
  { area: "stock-room", trade: "general", priority: "critical", status: "open", seeded: true, description: "Asbestos abatement — 1,500 SF black floor mastic (rear storage)", detail: "Confirmed ACM per Vertex Report #102403. Licensed abatement contractor required BEFORE any trade work. Do not disturb mastic until clearance received.", drawingRef: "ENV-01" },
  { area: "stock-room", trade: "general", priority: "high", status: "open", seeded: true, description: "Demo existing VCT tile and mastic (post-abatement only)", detail: "Wait for licensed industrial hygienist clearance after abatement. Save disposal manifests." },
  { area: "stock-room", trade: "general", priority: "normal", status: "open", seeded: true, description: "Install new VCT flooring per finish plan", detail: "12×12 VCT, Dollar Tree standard pattern. Coordinate adhesive with mastic removal.", drawingRef: "A-101" },
  { area: "stock-room", trade: "general", priority: "normal", status: "open", seeded: true, description: "Frame and drywall new partition walls per plan", detail: "5/8\" Type-X drywall to structure above ceiling where fire separation is required.", drawingRef: "A-101" },
  { area: "stock-room", trade: "general", priority: "normal", status: "open", seeded: true, description: "Install acoustical ceiling tile (ACT) grid and tiles", detail: "2×4 grid, standard ACT tiles. Install after all above-ceiling rough-in is inspected and approved.", drawingRef: "A-301" },
  { area: "stock-room", trade: "general", priority: "normal", status: "open", seeded: true, description: "Install HM door and frame — stockroom/sales floor", detail: "Per hardware schedule. Coordinate keying with Dollar Tree. Fire-rated assembly where required.", drawingRef: "A-801" },
  { area: "stock-room", trade: "general", priority: "normal", status: "open", seeded: true, description: "Concrete floor patching at all old penetrations", detail: "Use non-shrink grout. All old floor penetrations to be filled flush. Moisture test before VCT adhesive." },
  { area: "stock-room", trade: "electrical", priority: "high", status: "open", seeded: true, description: "Install LED industrial strip/high-bay lighting per reflected ceiling plan", detail: "Verify fixture schedule — confirm lumen output meets DT spec (min 30 FC maintained). Include emergency egress fixtures.", drawingRef: "E-101" },
  { area: "stock-room", trade: "electrical", priority: "critical", status: "open", seeded: true, description: "Install 20A GFCI duplex receptacles within 6ft of mop sink", detail: "NEC 210.8 — GFCI required for all receptacles within 6ft of any sink. Tamper-resistant, weather-rated device.", drawingRef: "E-101" },
  { area: "stock-room", trade: "electrical", priority: "high", status: "open", seeded: true, description: "Install low-voltage conduit for EMS/CYLON controls — RTU-1 zone", detail: "Run control wiring from CYLON panel to RTU-1 thermostat location per M-301. RFI #53 open — confirm controller model.", drawingRef: "E-301" },
  { area: "stock-room", trade: "electrical", priority: "high", status: "open", seeded: true, description: "Install exit and emergency egress lighting", detail: "Per life safety plan. Battery backup required on all units. Test at final inspection.", drawingRef: "E-201" },
  { area: "stock-room", trade: "electrical", priority: "normal", status: "open", seeded: true, description: "Label all stockroom circuits at panel — complete directory", detail: "Panel directory to be completed and laminated inside panel door. Coordinate with Seifert." },
  { area: "stock-room", trade: "plumbing", priority: "high", status: "open", seeded: true, description: "Install floor-mounted mop sink with wall-mounted faucet", detail: "Provide hot and cold supply with mixing valve. Min 3-inch drain. ADA accessible reach range.", drawingRef: "P-101" },
  { area: "stock-room", trade: "plumbing", priority: "high", status: "open", seeded: true, description: "Install floor drain (FD) at mop sink area", detail: "FD to sanitary drain. Trap primer required per WI SPS 382. Coordinate slope of floor to drain.", drawingRef: "P-101" },
  { area: "stock-room", trade: "plumbing", priority: "high", status: "open", seeded: true, description: "Install tempering/mixing valve — max 110°F at mop sink", detail: "WI SPS 382.41 — individual mixing valve required at mop sink if water heater set above 120°F.", drawingRef: "P-101" },
  { area: "stock-room", trade: "hvac", priority: "critical", status: "open", seeded: true, description: "RTU-1 (1,250 LB) — inspect existing roof curb before set", detail: "Same location W24. Inspect curb flashing, fasteners, and structural blocking. If deteriorated, replace before set. Coord: Chad DuFrane 920-731-5071.", drawingRef: "M-101" },
  { area: "stock-room", trade: "hvac", priority: "high", status: "open", seeded: true, description: "Install supply ductwork, diffusers, and return grilles per HVAC plan", detail: "Verify duct sizing matches schedule on M-101. Insulate per spec. Pressure test before ceiling close-in.", drawingRef: "M-101" },
  { area: "stock-room", trade: "hvac", priority: "high", status: "open", seeded: true, description: "Install CYLON EMS thermostat — RTU-1 zone", detail: "Schneider Electric CYLON controller. RFI #53 OPEN — EM-102/EM-103 still show SimpleStat/eSCi. Confirm correct controller model before ordering.", drawingRef: "M-301" },
  { area: "stock-room", trade: "hvac", priority: "high", status: "open", seeded: true, description: "Pressure test all ductwork before ceiling close-in", detail: "Duct leakage test per ASHRAE 90.1. Document results. Required before AHJ rough-in inspection." },
  { area: "stock-room", trade: "hvac", priority: "high", status: "open", seeded: true, description: "Commission RTU-1 startup — record airflow, temps, refrigerant pressures", detail: "Central Temp (Chad DuFrane 920-731-5071). Record static pressure, CFM at each diffuser, supply/return temps, refrigerant high/low side." },
  { area: "stock-room", trade: "fire", priority: "high", status: "open", seeded: true, description: "Install sprinkler heads per fire layout — ordinary hazard", detail: "Correct elevation and spacing per NFPA 13. Coordinate head locations with ceiling grid. Do not block with structure.", drawingRef: "FP-101" },
  { area: "stock-room", trade: "fire", priority: "high", status: "open", seeded: true, description: "Hydraulic calculation book approved by AHJ on file", detail: "Submit calc book to Marathon County AHJ. Do not install system until approved. Keep copy on site." },
  { area: "stock-room", trade: "fire", priority: "high", status: "open", seeded: true, description: "Flush, pressure test, and final inspection — coordinate Laser Fire and AHJ", detail: "Per NFPA 13 Section 14. Document all test results. Contact David Bartolerio (Laser Fire) for final scheduling." },

  // ── BATHROOMS ──────────────────────────────────────────────────────────────
  { area: "bathrooms", trade: "general", priority: "critical", status: "open", seeded: true, description: "Install slip-resistant ceramic tile floor (min 0.6 COF wet)", detail: "Verify COF certification from tile supplier. Slip-resistant grout joint. Document product data in closeout.", drawingRef: "A-401" },
  { area: "bathrooms", trade: "general", priority: "high", status: "open", seeded: true, description: "Install ceramic tile walls to height per finish plan", detail: "Typically 4'-0\" or full height per DT standard. Waterproof membrane (RedGard or equal) behind all tile in wet areas.", drawingRef: "A-401" },
  { area: "bathrooms", trade: "general", priority: "critical", status: "open", seeded: true, description: "Install ADA water closet — 17–19\" AFF seat height, elongated bowl", detail: "ADA §604.4. Centerline of WC 18\" from side wall per §604.2. Elongated bowl required for ADA.", drawingRef: "A-401" },
  { area: "bathrooms", trade: "general", priority: "critical", status: "open", seeded: true, description: "Install ADA lavatory — clear knee space below, rim max 34\" AFF", detail: "ADA §606.3. Insulate all exposed pipes under accessible lavatory to prevent contact burns.", drawingRef: "A-401" },
  { area: "bathrooms", trade: "general", priority: "critical", status: "open", seeded: true, description: "Install ADA grab bars — 42\" side wall, 36\" rear wall", detail: "ADA §604.5. Verify wall blocking is installed BEFORE drywall close-in. 1.5\" clearance from wall. 250 lb load capacity.", drawingRef: "A-401" },
  { area: "bathrooms", trade: "general", priority: "high", status: "open", seeded: true, description: "Install ADA accessories — TP holder, mirror (max 40\" AFF to bottom), soap/paper towel dispenser", detail: "ADA §603.3 for mirrors. All accessories within ADA reach range (max 48\" AFF). Coordinate mounting heights.", drawingRef: "A-401" },
  { area: "bathrooms", trade: "general", priority: "critical", status: "open", seeded: true, description: "Install HM door with lever hardware — min 32\" clear ADA opening", detail: "No knobs permitted per ADA. Closer must not exceed 5 lbf to open. Verify clear opening dimension.", drawingRef: "A-801" },
  { area: "bathrooms", trade: "general", priority: "high", status: "open", seeded: true, description: "Install occupancy/gender sign with Braille per WI accessibility code", detail: "Tactile characters and Braille required per ICC A117.1. Mount at 60\" AFF to centerline on latch side of door." },
  { area: "bathrooms", trade: "electrical", priority: "critical", status: "open", seeded: true, description: "Install GFCI duplex receptacle within 6ft of lavatory", detail: "NEC 210.8(A)(1) — GFCI required. Tamper-resistant device. Single-gang in tile — use proper tile ring adapter.", drawingRef: "E-101" },
  { area: "bathrooms", trade: "electrical", priority: "high", status: "open", seeded: true, description: "Install exhaust fan wired to switch or occupancy sensor", detail: "Min 50 CFM per ASHRAE 62.1. No recirculation. Exhaust must discharge to exterior. Coordinate with HVAC duct routing.", drawingRef: "E-101" },
  { area: "bathrooms", trade: "electrical", priority: "normal", status: "open", seeded: true, description: "Install light fixture — min 50 FC at mirror/task level", detail: "LED fixture. Verify illumination at mirror face. Install per reflected ceiling plan.", drawingRef: "E-101" },
  { area: "bathrooms", trade: "plumbing", priority: "critical", status: "open", seeded: true, description: "Install water closet — ADA rough-in, flush valve, angle stop", detail: "Centerline 18\" from side wall. Supply valve and shut-off required. Elongated bowl, ADA height.", drawingRef: "P-101" },
  { area: "bathrooms", trade: "plumbing", priority: "high", status: "open", seeded: true, description: "Install lavatory with P-trap, hot/cold supply, angle stops", detail: "ASSE 1069 angle stops. Insulate hot water supply pipe at accessible lavatory. P-trap must be removable.", drawingRef: "P-101" },
  { area: "bathrooms", trade: "plumbing", priority: "critical", status: "open", seeded: true, description: "Install tempering valve at lavatory — max 110°F", detail: "WI SPS 382.41 — individual mixing valve required. Factory set at 100–105°F. Verify temp after install.", drawingRef: "P-101" },
  { area: "bathrooms", trade: "plumbing", priority: "high", status: "open", seeded: true, description: "Install floor drain with trap seal primer in each restroom", detail: "Floor sloped 1/8\" per ft to drain. Trap primer per WI SPS 382 required on all floor drains.", drawingRef: "P-101" },
  { area: "bathrooms", trade: "plumbing", priority: "high", status: "open", seeded: true, description: "Vent all restroom fixtures per WI SPS 382 — AHJ inspection before wall close-in", detail: "Individual vent or wet vent per code. AHJ rough-in inspection required before closing walls." },
  { area: "bathrooms", trade: "hvac", priority: "high", status: "open", seeded: true, description: "Install exhaust fan and duct to exterior — restroom negative pressure required", detail: "Exhaust CFM must exceed supply CFM. Test and balance — document airflow differential. Laser Fire (David Bartolerio) for final.", drawingRef: "M-101" },
  { area: "bathrooms", trade: "hvac", priority: "normal", status: "open", seeded: true, description: "Install supply air diffuser (min 2 ACH supply air)", detail: "Size per manual D. Avoid direct throw onto water closet. Coordinate with ceiling tile layout.", drawingRef: "M-101" },
  { area: "bathrooms", trade: "fire", priority: "high", status: "open", seeded: true, description: "Install quick-response sprinkler heads per fire layout", detail: "Min 4\" from walls, 6\" from obstructions. Concealed or standard per ceiling type. Coordinate with ACT grid.", drawingRef: "FP-101" },

  // ── OFFICE ─────────────────────────────────────────────────────────────────
  { area: "office", trade: "general", priority: "normal", status: "open", seeded: true, description: "Frame and drywall office walls per plan", detail: "5/8\" Type X where fire separation is required. Verify fire-resistance ratings before framing.", drawingRef: "A-101" },
  { area: "office", trade: "general", priority: "normal", status: "open", seeded: true, description: "Install VCT or LVT flooring — per finish schedule", detail: "Coordinate with DT standard floor finish spec. Smooth substrate required for LVT.", drawingRef: "A-101" },
  { area: "office", trade: "general", priority: "normal", status: "open", seeded: true, description: "Install acoustical ceiling tile and grid to match adjacent spaces", detail: "Ceiling height per reflected ceiling plan. Confirm clearance above ceiling for mechanical." },
  { area: "office", trade: "general", priority: "normal", status: "open", seeded: true, description: "Install door, frame, and hardware per hardware schedule", detail: "Key to Dollar Tree master system. Verify hardware function. Door closer required if smoke partition.", drawingRef: "A-801" },
  { area: "office", trade: "electrical", priority: "normal", status: "open", seeded: true, description: "Install duplex receptacles per NEC 210.52 spacing (min 1 per wall)", detail: "20A circuits. NEC 210.52 — no point along wall more than 6ft from receptacle. AFCI protection per NEC 210.12.", drawingRef: "E-101" },
  { area: "office", trade: "electrical", priority: "normal", status: "open", seeded: true, description: "Install LED troffer or surface-mount lighting — min 50 FC at desk", detail: "Per reflected ceiling plan. Verify foot-candle levels. Occupancy sensor or switch per DT spec.", drawingRef: "E-101" },
  { area: "office", trade: "electrical", priority: "high", status: "open", seeded: true, description: "Install data/comm outlets — coordinate locations with Dollar Tree IT", detail: "Cat6 rough-in per DT IT spec. Confirm outlet count and locations with Dollar Tree PM before rough-in.", drawingRef: "E-101" },
  { area: "office", trade: "electrical", priority: "normal", status: "open", seeded: true, description: "Install occupancy sensor per Dollar Tree standard", detail: "Automatic-off occupancy sensor. No vacancy sensor. Install per E-101 reflected ceiling plan.", drawingRef: "E-101" },
  { area: "office", trade: "hvac", priority: "normal", status: "open", seeded: true, description: "Install supply air diffuser — office zone", detail: "Per reflected ceiling plan and HVAC schedule. Provide min 15 CFM/person outside air per ASHRAE 62.1.", drawingRef: "M-101" },
  { area: "office", trade: "hvac", priority: "normal", status: "open", seeded: true, description: "Verify office zone — separate scheduling from sales floor if required", detail: "If office requires different occupied/unoccupied schedule, confirm separate zone or CYLON setpoint override capability." },
  { area: "office", trade: "fire", priority: "normal", status: "open", seeded: true, description: "Install quick-response sprinkler head — office", detail: "Verify coverage per NFPA 13. Concealed or pendant per ceiling type. Document on fire plan.", drawingRef: "FP-101" },

  // ── STORE FLOOR ────────────────────────────────────────────────────────────
  { area: "store-floor", trade: "general", priority: "high", status: "open", seeded: true, description: "Concrete floor prep — grind, crack fill, patch all old penetrations", detail: "VCT requires flat, smooth substrate (max 3/16\" in 10ft). Moisture test required before adhesive. Old penetrations non-shrink grouted flush.", drawingRef: "A-101" },
  { area: "store-floor", trade: "general", priority: "high", status: "open", seeded: true, description: "Install new VCT flooring — 12×12 Dollar Tree standard layout", detail: "Coordinate layout drawing with DT. Install only after all slab penetrations, under-slab rough-in, and patching complete.", drawingRef: "A-101" },
  { area: "store-floor", trade: "general", priority: "normal", status: "open", seeded: true, description: "Install ACT grid and ceiling tiles — main sales area", detail: "2×4 grid per reflected ceiling plan. Verify heights match HVAC/electrical above-ceiling design. Install after all above-ceiling inspection.", drawingRef: "A-301" },
  { area: "store-floor", trade: "general", priority: "high", status: "open", seeded: true, description: "Install checkout counter/cashwrap per millwork plan", detail: "Coordinate with millwork contractor and Dollar Tree PM. ADA accessible checkout lane required per ADA §904.", drawingRef: "A-601" },
  { area: "store-floor", trade: "general", priority: "high", status: "open", seeded: true, description: "Install storefront entry vestibule glazing and automatic/power-assist doors", detail: "Aluminum storefront system. Tempered insulating glass. ADA-compliant threshold (max 1/2\" with bevel). Power-assist door if required.", drawingRef: "A-501" },
  { area: "store-floor", trade: "general", priority: "normal", status: "open", seeded: true, description: "Install gondola blocking/backing per plan before drywall close-in", detail: "3/4\" plywood blocking at all gondola wall-attachment points. Verify blocking heights match gondola model before framing.", drawingRef: "A-101" },
  { area: "store-floor", trade: "electrical", priority: "high", status: "open", seeded: true, description: "Install LED troffer fixtures per reflected ceiling plan — min 30 FC uniform", detail: "Verify fixture model per DT spec. Daylighting/daylight harvesting sensors per CYLON EMS where applicable.", drawingRef: "E-101" },
  { area: "store-floor", trade: "electrical", priority: "normal", status: "open", seeded: true, description: "Install quadplex receptacles at end caps and display columns", detail: "20A circuits to each end cap. Verify exact column/end cap positions with DT merchandising before rough-in.", drawingRef: "E-101" },
  { area: "store-floor", trade: "electrical", priority: "critical", status: "open", seeded: true, description: "RFI #53 OPEN — EM-102/EM-103 EMS controllers: SimpleStat/eSCi on drawings, NOT updated to CYLON", detail: "DO NOT rough-in EMS wiring until RFI #53 is resolved. Confirm correct CYLON controller model and I/O schedule with engineer before ordering.", drawingRef: "E-301" },
  { area: "store-floor", trade: "electrical", priority: "high", status: "open", seeded: true, description: "Install sub-panel and circuits for all cooler/freezer cases", detail: "Verify load calculations for all cooler cases. Dedicated breakers required. Coordinate panel schedule with Seifert (Kim Kluz 715-693-2625).", drawingRef: "E-201" },
  { area: "store-floor", trade: "electrical", priority: "high", status: "open", seeded: true, description: "Install POS/data conduit rough-in at all checkout positions", detail: "Coordinate exact POS positions with Dollar Tree IT. Provide floor sleeves if conduit must pass through slab.", drawingRef: "E-101" },
  { area: "store-floor", trade: "electrical", priority: "critical", status: "open", seeded: true, description: "Install exit and emergency egress lighting — full store per life safety plan", detail: "Battery backup on all units. Test and certify at final inspection. Photoluminescent path markings per IBC if required.", drawingRef: "E-201" },
  { area: "store-floor", trade: "electrical", priority: "critical", status: "open", seeded: true, description: "2×200A overhead service — main utility connections and meter", detail: "Seifert to coordinate with utility. Verify meter base location per utility requirement. Permit #26-013. Service entrance per E-001.", drawingRef: "E-001" },
  { area: "store-floor", trade: "hvac", priority: "critical", status: "open", seeded: true, description: "RTU-2 (1,250 LB) — NEW LOCATION W27, new roof curb and crane required", detail: "NEW LOCATION — new structural curb must be installed and flashed BEFORE unit set. Crane coordination with Central Temp (Chad DuFrane 920-731-5071). Rigging plan required.", drawingRef: "M-101" },
  { area: "store-floor", trade: "hvac", priority: "high", status: "open", seeded: true, description: "RTU-4 (755 LB) — same location W24, inspect existing curb before set", detail: "Inspect curb: corrosion, flashing condition, fasteners, structural blocking. Replace curb if deteriorated. Document inspection.", drawingRef: "M-101" },
  { area: "store-floor", trade: "hvac", priority: "high", status: "open", seeded: true, description: "XRTU-3 (existing to remain, W27) — test operation and document", detail: "Test unit before any close-in work. Record airflow, supply/return temps, refrigerant pressures. Tag unit if deficient.", drawingRef: "M-101" },
  { area: "store-floor", trade: "hvac", priority: "high", status: "open", seeded: true, description: "XRTU-5 (existing to remain) — test operation and document", detail: "Same as XRTU-3. Confirm serves correct zone per new layout. Record all test data. Report deficiencies to PM.", drawingRef: "M-101" },
  { area: "store-floor", trade: "hvac", priority: "high", status: "open", seeded: true, description: "Install supply diffusers and return grilles per HVAC duct schedule", detail: "Verify CFM values against schedule. Diffuser selection and placement per M-101. Coordinate with ceiling grid layout.", drawingRef: "M-101" },
  { area: "store-floor", trade: "hvac", priority: "critical", status: "open", seeded: true, description: "CYLON EMS commissioning — program all RTU zones per DT energy sequence", detail: "Schneider Electric CYLON. All zones must be programmed before CO inspection. RFI #53 — confirm EM-102/EM-103 controller model first.", drawingRef: "M-301" },
  { area: "store-floor", trade: "hvac", priority: "high", status: "open", seeded: true, description: "Test and balance all diffusers and return grilles — certified TAB report required", detail: "TAB contractor to provide certified report. File with project closeout documents. Required for CO." },
  { area: "store-floor", trade: "plumbing", priority: "normal", status: "open", seeded: true, description: "Install floor drain at janitor closet if on sales floor", detail: "FD with trap primer required. Slope floor 1/8\" per ft to drain. Coordinate with general contractor for correct elevation.", drawingRef: "P-101" },
  { area: "store-floor", trade: "fire", priority: "critical", status: "open", seeded: true, description: "Install sprinkler heads per fire layout — min 18\" clearance above top of highest storage", detail: "Ordinary hazard. Coordinate with gondola height. No obstruction to spray pattern. AHJ approval of layout required.", drawingRef: "FP-101" },
  { area: "store-floor", trade: "fire", priority: "critical", status: "open", seeded: true, description: "Install main fire riser, flow switch, tamper switches, and sectional valves", detail: "Tamper switches monitored at fire alarm panel. Flow switch to transmit waterflow alarm. Coordinate with fire alarm contractor.", drawingRef: "FP-001" },
  { area: "store-floor", trade: "fire", priority: "critical", status: "open", seeded: true, description: "Hydraulic calculations submitted to and approved by Marathon County AHJ", detail: "Do not install system until calc book is approved. Copy of approval letter to be kept on site." },
  { area: "store-floor", trade: "fire", priority: "high", status: "open", seeded: true, description: "Final fire sprinkler inspection with AHJ and Laser Fire", detail: "Contact David Bartolerio (Laser Fire) and Marathon County AHJ for inspection scheduling. Obtain signed inspection certificate." },

  // ── COOLERS / FREEZERS ─────────────────────────────────────────────────────
  { area: "coolers", trade: "general", priority: "high", status: "open", seeded: true, description: "Install walk-in cooler/freezer insulated panel system", detail: "Manufacturer-furnished panels. Verify R-value meets WI energy code. Install per manufacturer IOM. Do not damage panel vapor barrier.", drawingRef: "A-201" },
  { area: "coolers", trade: "general", priority: "high", status: "open", seeded: true, description: "Install insulated floor panel system in freezer area", detail: "Concrete sub-floor must be dry. Vapor barrier at slab before panel installation. Coordinate ramp/transition at entry.", drawingRef: "A-201" },
  { area: "coolers", trade: "general", priority: "high", status: "open", seeded: true, description: "Install insulated cooler/freezer doors with door heater strips", detail: "Freezer doors require heater strip to prevent frost sealing. Verify heater is wired and energized before startup.", drawingRef: "A-201" },
  { area: "coolers", trade: "general", priority: "critical", status: "open", seeded: true, description: "Seal ALL penetrations through cooler/freezer walls and floor — vapor barrier", detail: "Every penetration (pipe, conduit, refrigerant line) must be sealed with closed-cell foam backer and vapor barrier sealant. Critical in freezer." },
  { area: "coolers", trade: "electrical", priority: "critical", status: "open", seeded: true, description: "Install 3-phase power circuits for condensing units — size per equipment schedule", detail: "Confirm HP and amperage requirements with refrigeration contractor before rough-in. Dedicated circuits required. No shared neutrals.", drawingRef: "E-201" },
  { area: "coolers", trade: "electrical", priority: "high", status: "open", seeded: true, description: "Install vapor-proof LED lighting inside coolers and freezers", detail: "Min 20 FC inside cooler, 10 FC inside freezer. All fixtures must be wet-rated and vapor-tight. Verify T-rating for freezer service.", drawingRef: "E-101" },
  { area: "coolers", trade: "electrical", priority: "high", status: "open", seeded: true, description: "Install anti-sweat door heater circuits for all cooler/freezer doors", detail: "Heater circuits from dedicated breaker. Verify correct wattage per door model. Heater switch or EMS monitoring required.", drawingRef: "E-201" },
  { area: "coolers", trade: "electrical", priority: "high", status: "open", seeded: true, description: "Install CYLON EMS temperature monitoring points for each cooler/freezer zone", detail: "High/low temp alarms for each zone. Verify I/O point count with controls contractor. Alarm annunciation per DT requirements.", drawingRef: "E-301" },
  { area: "coolers", trade: "electrical", priority: "critical", status: "open", seeded: true, description: "GFCI protection on all cooler/freezer interior receptacles", detail: "NEC 210.8 — wet/damp location GFCI required on all interior receptacles. Use commercial-grade GFCI in wet-rated enclosure.", drawingRef: "E-201" },
  { area: "coolers", trade: "electrical", priority: "critical", status: "open", seeded: true, description: "Install condensing unit disconnects — within sight, within 50ft", detail: "NEC 440.14 — disconnect must be within line-of-sight and within 50ft of each condensing unit. Lockable in OFF position.", drawingRef: "E-201" },
  { area: "coolers", trade: "plumbing", priority: "high", status: "open", seeded: true, description: "Install condensate drain lines from evaporator coils — insulate", detail: "Route drain to approved floor drain. Insulate drain line to prevent sweating in ambient conditions. No air gaps in freezer.", drawingRef: "P-201" },
  { area: "coolers", trade: "plumbing", priority: "high", status: "open", seeded: true, description: "Install floor drains in cooler and freezer rooms", detail: "Floor drain with trap primer. Trap must be kept primed to prevent gas entry. Verify drain is not in the refrigeration vapor barrier zone.", drawingRef: "P-201" },
  { area: "coolers", trade: "plumbing", priority: "high", status: "open", seeded: true, description: "Heat trace condensate drain lines in freezer areas to prevent freezing", detail: "Self-regulating heat trace tape on any condensate drain line exposed to below-freezing conditions. Plug into dedicated circuit." },
  { area: "coolers", trade: "hvac", priority: "high", status: "open", seeded: true, description: "Install evaporator coil/air units — hang per manufacturer, coordinate refrigerant penetrations", detail: "Hang unit per manufacturer IOM. Coordinate all refrigerant line and drain penetrations through panel walls with general contractor.", drawingRef: "M-201" },
  { area: "coolers", trade: "hvac", priority: "critical", status: "open", seeded: true, description: "Install ACR copper refrigerant piping — braze with nitrogen purge, insulate suction line", detail: "ASTM B280 ACR copper. All joints brazed with dry nitrogen purge flowing. Suction line insulated full length. No mechanical joints.", drawingRef: "M-201" },
  { area: "coolers", trade: "hvac", priority: "critical", status: "open", seeded: true, description: "Pressure test refrigeration circuit per ASHRAE 15 — nitrogen, document results", detail: "Nitrogen pressure test to 150% of high-side design pressure. Record test pressure, time, and final pressure with timestamps. No loss." },
  { area: "coolers", trade: "hvac", priority: "critical", status: "open", seeded: true, description: "Commission refrigeration: triple evacuate to 300 microns, charge refrigerant, record superheat/subcooling", detail: "Triple evacuation with electronic micron gauge. Charge per nameplate weight. Record refrigerant type, charge, superheat, subcooling at startup.", drawingRef: "M-201" },
  { area: "coolers", trade: "fire", priority: "critical", status: "open", seeded: true, description: "Install freezer-rated sprinkler heads — dry pendant or CPVC antifreeze system", detail: "Standard wet-pipe heads CANNOT be used in freezer areas. Dry pendant heads or CPVC antifreeze system per NFPA 13. AHJ approval required.", drawingRef: "FP-201" },
  { area: "coolers", trade: "fire", priority: "high", status: "open", seeded: true, description: "Verify sprinkler head coverage and spacing per NFPA 13 in cooler/freezer", detail: "Confirm head spacing, obstruction clearances, and coverage areas per hydraulic calculations. Coordinate with shelving layout.", drawingRef: "FP-201" },

  // ── EXTERIOR ───────────────────────────────────────────────────────────────
  { area: "exterior", trade: "general", priority: "high", status: "open", seeded: true, description: "Install storefront glazing system and automatic/power-assist entry doors", detail: "Aluminum curtainwall/storefront per plan. Tempered insulating glass. Thermal break per energy code. Verify air infiltration rating.", drawingRef: "A-501" },
  { area: "exterior", trade: "general", priority: "critical", status: "open", seeded: true, description: "Install ADA accessible path from parking to entrance — max 1:20 slope (ramp if steeper)", detail: "Detectable warning surface required at all curb cuts. If slope exceeds 1:20 provide ramp with handrails. Document compliance.", drawingRef: "A-101" },
  { area: "exterior", trade: "general", priority: "normal", status: "open", seeded: true, description: "Install removable bollards at main entrance — powder-coated yellow", detail: "Lockable anchor socket recessed in slab. Bollards to protect storefront from vehicle intrusion per DT standard.", drawingRef: "A-101" },
  { area: "exterior", trade: "general", priority: "high", status: "open", seeded: true, description: "Install signage backing and blocking per sign plan", detail: "Coordinate with sign contractor on size and connection requirements. Structural blocking in wall for wall-mounted signs before close-in." },
  { area: "exterior", trade: "general", priority: "normal", status: "open", seeded: true, description: "Repair exterior masonry or EIFS as required — match existing texture and color", detail: "Document all damaged areas before repair. EIFS repairs by qualified applicator. Manufacturer color match required.", drawingRef: "A-201" },
  { area: "exterior", trade: "general", priority: "normal", status: "open", seeded: true, description: "Install downspouts, splash blocks, and verify site drainage — no ponding", detail: "All roof drain leaders to connect to site drainage. Positive drainage away from building per civil grading plan." },
  { area: "exterior", trade: "electrical", priority: "high", status: "open", seeded: true, description: "Install exterior canopy/soffit LED lighting — photocell controlled", detail: "Photocell required. Verify lumen output meets DT exterior spec. Coordinate fixture locations with canopy/sign contractor.", drawingRef: "E-101" },
  { area: "exterior", trade: "electrical", priority: "high", status: "open", seeded: true, description: "Install pylon/monument sign power circuit", detail: "Verify circuit ampacity with sign contractor. GFCI protection required if accessible receptacle at sign base.", drawingRef: "E-101" },
  { area: "exterior", trade: "electrical", priority: "high", status: "open", seeded: true, description: "Install exterior GFCI duplex receptacles with weatherproof in-use covers", detail: "NEC 406.9 — weatherproof in-use covers required on all exterior receptacles. GFCI protected per NEC 210.8(A).", drawingRef: "E-101" },
  { area: "exterior", trade: "electrical", priority: "critical", status: "open", seeded: true, description: "Install emergency egress lighting at all exterior exit doors", detail: "Battery backup units at each exit. Test at final inspection. Photoluminescent exit signs where required by IBC.", drawingRef: "E-201" },
  { area: "exterior", trade: "plumbing", priority: "critical", status: "open", seeded: true, description: "Install RPBA backflow preventer on domestic water service entry", detail: "WI SPS 382 and AWWA — Reduced Pressure Backflow Assembler required at service entry. AHJ testing and approval required annually.", drawingRef: "P-001" },
  { area: "exterior", trade: "plumbing", priority: "high", status: "open", seeded: true, description: "Verify roof drain leaders connect properly to site drainage system", detail: "Inspect all roof drain leader connections. Ensure positive drainage away from building. No ponding within 10ft of foundation." },
  { area: "exterior", trade: "plumbing", priority: "high", status: "open", seeded: true, description: "Coordinate utility service taps with City of Rothschild (Permit #26-013)", detail: "Confirm service tap permits, inspection windows, and utility shutdown scheduling with city public works. Best1 (Zach/Nate Roth 715-241-0883)." },
  { area: "exterior", trade: "hvac", priority: "critical", status: "open", seeded: true, description: "RTU-2 (1,250 LB) — set at NEW LOCATION W27, new structural roof curb required", detail: "CRITICAL: New curb must be installed, inspected, and flashed BEFORE crane day. Coordinate crane and rigging plan with Central Temp (Chad DuFrane 920-731-5071).", drawingRef: "M-101" },
  { area: "exterior", trade: "hvac", priority: "critical", status: "open", seeded: true, description: "Install new structural roof curb for RTU-2 at W27 — flash and waterproof", detail: "Coordinate with structural engineer if additional blocking needed. Flash per roofing spec. No roof penetration until curb is inspected.", drawingRef: "M-101" },
  { area: "exterior", trade: "hvac", priority: "high", status: "open", seeded: true, description: "RTU-1 (W24) — inspect and confirm existing curb condition, replace if deteriorated", detail: "Inspect curb: corrosion, flashing integrity, fasteners, structural blocking adequacy. Document findings. Replace curb if any deficiency.", drawingRef: "M-101" },
  { area: "exterior", trade: "hvac", priority: "high", status: "open", seeded: true, description: "Connect refrigerant lines and CYLON control wiring to RTU-2 after set", detail: "All refrigerant connections after unit is set and secured. CYLON control wiring per M-301. Pressure test before startup.", drawingRef: "M-101" },
  { area: "exterior", trade: "hvac", priority: "normal", status: "open", seeded: true, description: "Install equipment screens or parapet rails if required by municipality or Dollar Tree", detail: "Confirm requirements with AHJ and Dollar Tree PM. Coordinate screen attachment with structural and roofing." },
  { area: "exterior", trade: "fire", priority: "critical", status: "open", seeded: true, description: "Install Knox Box at main entry — keyed per Marathon County AHJ requirement", detail: "Contact Marathon County/Wausau FD for key specification and required location. Mount at 4'-6\" AFF maximum. Install before CO inspection.", drawingRef: "FP-001" },
  { area: "exterior", trade: "fire", priority: "critical", status: "open", seeded: true, description: "Install fire department connection (FDC) — 3\" siamese at accessible location", detail: "Location must be approved by AHJ. Breakaway caps and FDC sign required per NFPA 13. Coordinate connection to sprinkler riser.", drawingRef: "FP-001" },
  { area: "exterior", trade: "fire", priority: "critical", status: "open", seeded: true, description: "Install DCVA/RPBA backflow preventer on fire sprinkler service", detail: "AHJ to specify assembly type. Inspect and test per NFPA 25 annual testing. Record on backflow test report.", drawingRef: "FP-001" },
  { area: "exterior", trade: "fire", priority: "critical", status: "open", seeded: true, description: "Schedule and complete fire department final with Marathon County AHJ", detail: "Coordinate final inspection with Marathon County AHJ and Laser Fire. CO cannot be issued without fire system acceptance." },
];

async function seedDatabase() {
  const existingProjects = await storage.getProjects();
  const hasRothschild = existingProjects.some(p => p.name.includes("Rothschild"));
  if (existingProjects.length === 0) {
    const p1 = await storage.createProject({
      name: "Highland Heights School",
      location: "Denver, CO",
      description: "New elementary school construction.",
    });
    await storage.createDocument(p1.id, {
      name: "Architectural Blueprint V1",
      type: "blueprint",
      trade: "general",
      content: "Main level blueprint showing all rooms.",
    });
    await storage.createDailyLog(p1.id, {
      trade: "plumbing",
      date: new Date().toISOString().split('T')[0],
      completedWork: "Installed main water line in East Wing.",
      discrepancies: "Pipes clash with HVAC duct location.",
    });
    await storage.createRfi(p1.id, {
      trade: "plumbing",
      title: "HVAC Duct Clash",
      description: "Plumbing lines on East wing intersect with planned HVAC ducts. Need clarification on rerouting.",
      status: "open",
    });
  }

  if (!hasRothschild) {
    const p = await storage.createProject({
      name: "DT-Rothschild HVAC Project",
      location: "Rothschild, WI",
      description: "Full construction renovation — all trades on site. Location: 1029 E Grand Ave, Rothschild, WI. Dollar Tree Store #10746. Final construction documents dated 10/30/20. Trades include HVAC, electrical, plumbing, fire protection, and general construction per applicable Wisconsin SPS codes and national standards.",
    });
    await storage.createDocument(p.id, {
      name: "Final Construction Documents 10.30.20",
      type: "blueprint",
      trade: "general",
      content: "Blueprint uploaded. Click 'Analyze with AI' to perform a full multi-trade code review and discrepancy scan. Note: Upload blueprint pages as PNG/JPG images for full AI vision analysis.",
    });
    await storage.createDocument(p.id, {
      name: "Environmental Report — Asbestos Survey (Version 1)",
      type: "inspection",
      trade: "general",
      content: `ASBESTOS IDENTIFICATION SURVEY — VERTEX COMPANIES, LLC
Vertex Project No: 102403 | Dollar Tree Store #10746
Property: 1029 E Grand Ave, Rothschild, WI
Prepared For: Dollar Tree Stores, Inc., Chesapeake, VA — Attn: Steven McMahon
Inspection Date: January 23, 2025 | Report Date: January 28, 2025
Inspector: Tucker Ryckman — Licensed WI Asbestos Building Inspector No. AII-253634
Report Reviewed By: Donald P. Heim, VP — The Vertex Companies, LLC

=== CONFIRMED ACM (ASBESTOS-CONTAINING MATERIALS) ===
Location: Rear storage area
Material: Black mastic under beige tile
Estimated Quantity: 1,500 SF
Friable: NO | Condition: GOOD | Debris: NO

NOTE: Black floor mastic was confirmed asbestos-containing. If observed in concealed locations during renovation, treat as PACM (Presumed Asbestos-Containing Material).

=== KEY RESTRICTIONS ===
- Survey limited to exposed materials only
- No inspection inside enclosed walls/ceilings, fire doors, HVAC equipment, or permanent structures
- No roofing materials inspected
- No below-grade or stored materials inspected
- All concealed suspect materials should be treated as PACM

=== CONCLUSIONS & REQUIREMENTS ===
ACM FOUND: Yes — black floor mastic (rear storage area)
ACM materials impacted by renovation MUST be removed FIRST by a licensed asbestos abatement contractor per all applicable asbestos regulations before any renovation work begins.
If additional suspect materials found during renovation: halt work, collect samples, analyze before disturbing.

=== REGULATORY BASIS ===
OSHA 29 CFR 1926.1101 (Asbestos Construction Standard)
EPA 40 CFR 61 Subpart M (NESHAP)
EPA 40 CFR 763 Subpart E (AHERA)
PLM Analysis per EPA Method 600R/R-93/116 by Eckhart Environmental Services, LLC (NIST Accredited, Lab Code 600273-0)`,
    });
    await storage.createRfi(p.id, {
      trade: "hvac",
      title: "Blueprint PDF Encoding Issue — Upload Pages as Images for AI Analysis",
      description: "The uploaded PDF has internal encoding (Marked Content) that prevents direct text extraction. To enable full AI vision analysis across all trades, please upload individual blueprint pages as PNG or JPG images using the Documents tab. The AI will read and analyze each sheet visually for code compliance, duct sizing, equipment placement, and discrepancies.",
      status: "open",
    });
    await storage.createRfi(p.id, {
      trade: "general",
      title: "CRITICAL: Asbestos Abatement Required Before All Renovation Work",
      description: "Confirmed ACM identified: Black floor mastic under beige tile in rear storage area (1,500 SF). Per Vertex Environmental Report (Project #102403, Jan 2025), this material MUST be removed by a licensed asbestos abatement contractor before any renovation activities begin. No trade work may disturb this area until abatement is complete and clearance is received. Inspector: Tucker Ryckman, WI License AII-253634.",
      status: "open",
    });
    await storage.createRfi(p.id, {
      trade: "general",
      title: "Asbestos PACM — All Concealed Black Mastic Must Be Treated as Presumed ACM",
      description: "Black floor mastic confirmed asbestos-containing (1,500 SF, rear storage). If black mastic is found in ANY concealed location during renovation, all trades must halt work immediately, notify the PM, and do not disturb until sampled and cleared. Treat as PACM per OSHA 29 CFR 1926.1101.",
      status: "open",
    });
    await storage.createRfi(p.id, {
      trade: "general",
      title: "Environmental Survey Gaps — Walls, Ceilings, Roof, HVAC Equipment Not Inspected",
      description: "The Vertex asbestos survey only covered exposed materials. The following were NOT inspected: enclosed walls/ceilings, fire doors, electrical equipment, HVAC equipment, roofing, and below-grade areas. Coordinate with Dollar Tree (Steven McMahon) before trade work in these areas. Additional ACM sampling may be required.",
      status: "open",
    });
  }

  // Seed punchlist for Rothschild project regardless of whether it was just created
  const allProjects = await storage.getProjects();
  const rothschild = allProjects.find(p => p.name.includes("Rothschild"));
  if (rothschild) {
    const seededCount = await storage.countSeededPunchlistItems(rothschild.id);
    if (seededCount === 0) {
      console.log("[seed] Seeding punchlist items for DT10746 Rothschild...");
      for (const item of PUNCHLIST_SEED) {
        await storage.createPunchlistItem({ ...item, projectId: rothschild.id });
      }
      console.log(`[seed] Seeded ${PUNCHLIST_SEED.length} punchlist items.`);
    }
  }

}

export async function registerRoutes(
  httpServer: Server,
  app: Express
): Promise<Server> {

  // Health check — responds immediately, no DB required
  app.get("/health", (_req, res) => res.json({ status: "ok" }));

  // Source code download (no auth required — URL is not published anywhere)
  app.get("/download-source", (_req, res) => {
    const fs = require("fs");
    const path = require("path");
    const zipPath = path.join(process.cwd(), "buildmind-ai.zip");
    if (!fs.existsSync(zipPath)) {
      return res.status(404).json({ message: "Zip not found" });
    }
    res.setHeader("Content-Disposition", "attachment; filename=buildmind-ai.zip");
    res.setHeader("Content-Type", "application/zip");
    fs.createReadStream(zipPath).pipe(res);
  });

  // Auth
  app.post("/api/auth/login", (req, res) => {
    const { password } = req.body;
    const appPassword = process.env.APP_PASSWORD;
    if (!appPassword) {
      console.error("[auth] APP_PASSWORD not set");
      return res.status(500).json({ message: "Server not configured" });
    }
    if (password?.trim() === appPassword?.trim()) {
      req.session.authenticated = true;
      return req.session.save((err) => {
        if (err) {
          console.error("[auth] session.save error:", err);
          return res.status(500).json({ message: "Session error" });
        }
        console.log("[auth] login success, session id:", req.session.id);
        return res.json({ success: true });
      });
    }
    console.warn("[auth] login failed — password mismatch");
    return res.status(401).json({ message: "Incorrect password" });
  });

  app.post("/api/auth/logout", (req, res) => {
    req.session.destroy(() => {
      res.json({ success: true });
    });
  });

  app.get("/api/auth/status", (req, res) => {
    res.json({ authenticated: req.session.authenticated === true });
  });

  // Protect all other API routes
  app.use("/api", (req, res, next) => {
    if (req.path.startsWith("/auth/")) return next();
    if (req.path.startsWith("/reports/")) return next();
    if (req.session.authenticated) return next();
    return res.status(401).json({ message: "Not authenticated" });
  });

  // Projects
  app.get(api.projects.list.path, async (req, res) => {
    res.json(await storage.getProjects());
  });

  app.get(api.projects.get.path, async (req, res) => {
    const project = await storage.getProject(Number(req.params.id));
    if (!project) return res.status(404).json({ message: "Project not found" });
    res.json(project);
  });

  app.post(api.projects.create.path, async (req, res) => {
    try {
      const input = api.projects.create.input.parse(req.body);
      res.status(201).json(await storage.createProject(input));
    } catch (err) {
      if (err instanceof z.ZodError)
        return res.status(400).json({ message: err.errors[0].message, field: err.errors[0].path.join('.') });
      const message = err instanceof Error ? err.message : "Internal server error";
      return res.status(500).json({ message });
    }
  });

  app.patch("/api/projects/:id", async (req, res) => {
    try {
      const project = await storage.updateProject(Number(req.params.id), req.body);
      res.json(project);
    } catch (err: any) {
      res.status(500).json({ message: err.message });
    }
  });

  // Documents
  app.get(api.documents.list.path, async (req, res) => {
    const docs = await storage.getDocuments(Number(req.params.projectId));
    // Strip fileData from list to keep payload small
    res.json(docs.map(d => ({ ...d, fileData: d.fileData ? "[file stored]" : null })));
  });

  app.post(api.documents.create.path, async (req, res) => {
    try {
      const input = api.documents.create.input.parse(req.body);
      const { fileBase64, ...rest } = input;

      // Detect mime type from base64 header or extension
      let fileMime: string | undefined;
      if (fileBase64) {
        if (fileBase64.startsWith('/9j/') || fileBase64.startsWith('iVBORw')) fileMime = 'image/jpeg';
        else if (fileBase64.startsWith('iVBOR')) fileMime = 'image/png';
        else if (fileBase64.startsWith('JVBER')) fileMime = 'application/pdf';
        else fileMime = 'image/jpeg'; // default assumption
      }

      const doc = await storage.createDocument(Number(req.params.projectId), {
        ...rest,
        content: fileBase64
          ? "Analyzing document…"
          : (rest.content || "Document saved."),
        fileData: fileBase64 || undefined,
        fileMime,
      });
      res.status(201).json({ ...doc, fileData: doc.fileData ? "[file stored]" : null });

      // Fire-and-forget: auto-analyze any uploaded file in the background
      if (fileBase64) {
        autoAnalyzeDocument(doc.id, Number(req.params.projectId)).catch(() => {});
      }
    } catch (err) {
      if (err instanceof z.ZodError)
        return res.status(400).json({ message: err.errors[0].message, field: err.errors[0].path.join('.') });
      const message = err instanceof Error ? err.message : "Internal server error";
      return res.status(500).json({ message });
    }
  });

  // Analyze document with AI vision
  app.post(api.documents.analyze.path, async (req, res) => {
    try {
      const doc = await storage.getDocument(Number(req.params.id));
      if (!doc) return res.status(404).json({ message: "Document not found" });

      const [project, openRfisForVision, priorAnalyses] = await Promise.all([
        storage.getProject(Number(req.params.projectId)),
        storage.getRfis(Number(req.params.projectId)),
        storage.getBlueprintAnalyses(Number(req.params.projectId)),
      ]);
      const location = project?.location || "Unknown location";
      const trade = doc.trade || "general";
      const codes = TRADE_CODES[trade] || TRADE_CODES.general;
      const openRfiList = openRfisForVision.filter(r => r.status === 'open').slice(0, 20)
        .map(r => `- [${r.rfiNumber || r.id}] ${r.trade.toUpperCase()}: ${r.title}`).join('\n');
      const priorAnalysisSummary = priorAnalyses.slice(-3)
        .map(a => `- ${a.sheetName || a.id}: ${a.summary?.slice(0, 150) || ''}`).join('\n');

      const systemPrompt = `You are an expert construction code inspector and ${trade.toUpperCase()} engineer analyzing blueprints for Dollar Tree Store #10746 (Store #45117), 1029 E Grand Ave, Rothschild, WI 54476.
Trade scope: ${trade.toUpperCase()} | Location: ${location}
Applicable codes: ${codes}

AUTHORITATIVE RTU/HVAC INVENTORY (cross-reference on roof/mechanical plans):
RTU-1  | 1,250 LBS | REPLACED — new unit | Existing location | W24 STL. BEAM + DBL. JOIST
RTU-2  | 1,250 LBS | REPLACED — new unit | NEW LOCATION (moved) | W27 STL. BEAM + DBL. JOIST — 4 trades affected
XRTU-3 | STAYS — service only | W27 STL. BEAM (Δ2 cloud on S2)
RTU-4  | 755 LBS | REPLACED — new unit | Existing location | W24 STL. BEAM + DBL. JOIST (Δ1 cloud on S2)
XRTU-5 | STAYS — service only
EMS: CYLON by Schneider Electric — EM-102/EM-103 still show legacy SimpleStat/eSCi (NOT updated in REV 4)

OPEN RFIs TO CROSS-REFERENCE WHILE READING THIS DRAWING:
${openRfiList || "None"}

PRIOR BLUEPRINT ANALYSES ON FILE:
${priorAnalysisSummary || "None yet"}

Analyze the provided document. Structure your response:
1. **Document Summary** — What you see in this drawing/document
2. **Trade Scope of Work** — Detailed ${trade.toUpperCase()} work required
3. **Code Compliance Review** — Check against codes above, cite specific sections
4. **Discrepancies Found** — Conflicts, clashes, violations, missing specs, or items that conflict with any open RFI above
5. **RFI Recommendations** — Items needing clarification, especially anything touching RTU-2 relocation or EMS wiring
6. **Inspection Checklist** — What the ${trade.toUpperCase()} inspector will verify
7. **Correction Guidance** — How to fix violations found`;

      let analysisResult: string;

      if (doc.fileData) {
        // Use vision API to read the actual image/document
        const isImage = doc.fileMime?.startsWith('image/');
        
        if (isImage) {
          // Use GPT vision for actual image analysis
          const gpt = getGptClient();
          const response = await gpt.chat.completions.create({
            model: "gpt-4o",
            max_tokens: 8192,
            messages: [
              { role: "system", content: systemPrompt },
              {
                role: "user",
                content: [
                  {
                    type: "image_url",
                    image_url: {
                      url: `data:${doc.fileMime};base64,${doc.fileData}`,
                      detail: "high"
                    }
                  },
                  {
                    type: "text",
                    text: `Analyze this ${trade.toUpperCase()} blueprint/document: "${doc.name}". Provide a comprehensive code compliance review and identify ALL discrepancies.`
                  }
                ]
              }
            ],
          });
          analysisResult = response.choices[0]?.message?.content || "No analysis generated.";
        } else if (doc.fileMime === "application/pdf") {
          // PDF — extract real text content first, then analyze
          const pdfText = await extractPdfText(doc.fileData);
          const gpt = getGptClient();
          const response = await gpt.chat.completions.create({
            model: "gpt-4o",
            max_tokens: 8192,
            messages: [
              { role: "system", content: systemPrompt },
              {
                role: "user",
                content: pdfText
                  ? `Document: "${doc.name}" — Trade: ${trade.toUpperCase()}\n\nFULL PDF CONTENT:\n${pdfText.slice(0, 14000)}\n\nAnalyze this document completely. Extract all key details, specifications, dates, amounts, contacts, and code compliance items.`
                  : `Document: "${doc.name}" — Trade: ${trade.toUpperCase()}\n\nThis PDF appears to be a scanned image without extractable text. Based on the document name and trade scope, provide a comprehensive ${trade.toUpperCase()} analysis and checklist for this project.\n\nNote: For scanned PDFs, upload pages as individual JPEG/PNG images for full visual AI analysis.`
              }
            ],
          });
          analysisResult = response.choices[0]?.message?.content || "No analysis generated.";
        } else {
          analysisResult = "Unsupported file type. Please upload a JPEG, PNG, or PDF file.";
        }
      } else {
        // Text-based analysis using document content via GPT-4o
        const gpt = getGptClient();
        const response = await gpt.chat.completions.create({
          model: "gpt-4o",
          max_tokens: 8192,
          messages: [
            { role: "system", content: systemPrompt },
            {
              role: "user",
              content: `Document: "${doc.name}"\nTrade: ${trade.toUpperCase()}\nLocation: ${location}\nContent: ${doc.content || "No content available."}\n\nProvide a full code compliance review and identify any discrepancies.`
            }
          ],
        });
        analysisResult = response.choices[0]?.message?.content || "No analysis generated.";
      }

      // Save analysis back to document content
      await storage.updateDocumentContent(doc.id, analysisResult);

      // Auto-create RFIs for any critical issues found
      const rfiMatches = analysisResult.match(/(?:RFI|Issue|Discrepancy|Violation)[^\n]*:[^\n]+/gi) || [];
      if (rfiMatches.length > 0 && rfiMatches.length <= 5) {
        for (const rfiText of rfiMatches.slice(0, 3)) {
          await storage.createRfi(Number(req.params.projectId), {
            trade,
            title: rfiText.slice(0, 80).replace(/^(RFI|Issue|Discrepancy|Violation)\s*#?\d*:\s*/i, "").trim(),
            description: `Auto-generated from AI analysis of "${doc.name}". ${rfiText}`,
            status: "open",
          });
        }
      }

      res.json({ message: analysisResult });
    } catch (error) {
      console.error("Analyze error:", error);
      res.status(500).json({ message: "Failed to analyze document. Please try again." });
    }
  });

  // RFIs
  app.get(api.rfis.list.path, async (req, res) => {
    res.json(await storage.getRfis(Number(req.params.projectId)));
  });

  app.post(api.rfis.create.path, async (req, res) => {
    try {
      const input = api.rfis.create.input.parse(req.body);
      res.status(201).json(await storage.createRfi(Number(req.params.projectId), input));
    } catch (err) {
      if (err instanceof z.ZodError)
        return res.status(400).json({ message: err.errors[0].message, field: err.errors[0].path.join('.') });
      const message = err instanceof Error ? err.message : "Internal server error";
      return res.status(500).json({ message });
    }
  });

  // Daily Logs
  app.get(api.dailyLogs.list.path, async (req, res) => {
    res.json(await storage.getDailyLogs(Number(req.params.projectId)));
  });

  app.post(api.dailyLogs.create.path, async (req, res) => {
    try {
      const input = api.dailyLogs.create.input.parse(req.body);
      const log = await storage.createDailyLog(Number(req.params.projectId), {
        trade: input.trade,
        date: input.date,
        completedWork: input.completedWork,
        discrepancies: input.discrepancies,
        photoUrl: input.photoBase64 ? `data:image/jpeg;base64,${input.photoBase64}` : undefined
      });
      res.status(201).json(log);
    } catch (err) {
      if (err instanceof z.ZodError)
        return res.status(400).json({ message: err.errors[0].message, field: err.errors[0].path.join('.') });
      const message = err instanceof Error ? err.message : "Internal server error";
      return res.status(500).json({ message });
    }
  });

  // Chats — AI assistant with full project context
  app.get(api.chats.list.path, async (req, res) => {
    res.json(await storage.getChats(Number(req.params.projectId)));
  });

  app.post(api.chats.create.path, async (req, res) => {
    try {
      const input = api.chats.create.input.parse(req.body);
      const projectId = Number(req.params.projectId);

      await storage.createChat(projectId, { role: "user", content: input.content });

      const [project, docs, rfis, logs, history, insights, punchItems, events, submittals, contractors] = await Promise.all([
        storage.getProject(projectId),
        storage.getDocuments(projectId),
        storage.getRfis(projectId),
        storage.getDailyLogs(projectId),
        storage.getChats(projectId),
        storage.getInsights(projectId, 3),
        storage.getPunchlistItems(projectId),
        storage.getFieldEvents(projectId),
        storage.getSubmittals(projectId),
        storage.getTradeContractors(projectId),
      ]);

      const trade = docs[0]?.trade || "general";
      const codes = TRADE_CODES[trade] || TRADE_CODES.general;
      const docSummary = docs.slice(0, 12).map(d => `- ${d.name} (${d.type}, ${d.trade})`).join('\n');
      const openRfis = rfis.filter(r => r.status === 'open').slice(0, 25);
      const rfiSummary = openRfis.map(r => `- [${r.rfiNumber || r.id}] ${r.trade.toUpperCase()}: ${r.title}`).join('\n');
      const logSummary = logs.slice(-5).map(l => `- [${l.date}] ${l.trade}: ${l.completedWork}${l.discrepancies ? ' | Issues: ' + l.discrepancies : ''}`).join('\n');
      const insightSummary = insights.length > 0
        ? insights.map((ins) => `[Loop ${ins.loopNumber}] ${ins.content}`).join('\n\n')
        : "No learning insights generated yet — loop runs every 24 minutes.";

      // Punchlist summary by trade
      const openPunch = punchItems.filter(p => p.status === 'open' || p.status === 'in-progress');
      const criticalPunch = punchItems.filter(p => p.priority === 'critical' && p.status !== 'complete');
      const punchByTrade: Record<string, number> = {};
      openPunch.forEach(p => { punchByTrade[p.trade] = (punchByTrade[p.trade] || 0) + 1; });
      const punchSummary = [
        `Total open: ${openPunch.length} | Critical: ${criticalPunch.length} | Complete: ${punchItems.filter(p => p.status === 'complete').length}`,
        Object.entries(punchByTrade).map(([t, n]) => `${t.toUpperCase()}: ${n} open`).join(' | '),
        criticalPunch.slice(0, 5).map(p => `⚠ [${p.area}/${p.trade}] ${p.description}`).join('\n'),
      ].filter(Boolean).join('\n');

      // Field events (last 8 — issues, safety, delays)
      const recentEvents = events.slice(-8).map(e =>
        `[${e.type?.toUpperCase() || 'EVENT'}][${e.severity || 'med'}][${e.date || ''}] ${e.title}: ${e.description?.slice(0, 120) || ''}`
      ).join('\n');

      // Submittals
      const pendingSubmittals = submittals.filter(s => s.status !== 'approved' && s.status !== 'rejected');
      const submittalSummary = pendingSubmittals.slice(0, 10).map(s =>
        `- [${s.status?.toUpperCase() || 'PENDING'}] ${s.trade?.toUpperCase()}: ${s.name}${s.dueDate ? ' — Due: ' + s.dueDate : ''}`
      ).join('\n');

      // Contractor payment status (live from DB)
      const contractorSummary = contractors.map(c =>
        `${c.trade?.toUpperCase()}: ${c.companyName} | ${c.contactPhone || ''} | $${c.contractAmount || 'TBD'} | ${c.paymentTerms || 'Net 30'}${c.notes ? ' | ' + c.notes.slice(0, 60) : ''}`
      ).join('\n');

      const systemMessage = `You are BuildMind AI, an expert construction superintendent and code compliance assistant. You serve as the field intelligence system for the active job described below. You have hardcoded authoritative project knowledge plus live data from the project database — use both together.

══════════════════════════════════════════════════════════════
ACTIVE PROJECT — AUTHORITATIVE KNOWLEDGE (always current)
══════════════════════════════════════════════════════════════
PROJECT: Dollar Tree Store #10746 (Store #45117)
ADDRESS: 1029 E Grand Ave, Rothschild, WI 54476
PERMIT: #26-013 | Fee paid: $2,517.76
DSPS: CB-012600013-PRBH (approved 1/13/2026, expires 1/13/2028)
CONSTRUCTION: 03/09/2026 – 04/17/2026 (~6 weeks)
BUILDING: 11,252 SF | 1 Story | Type II-B | Fully Sprinklered
OCCUPANCY: M-Mercantile | 156 persons (152 sales + 4 stockroom)
EST. COST: $377,000 | Subcontracts: ~$127,678
ELECTRICAL SERVICE: 2 × 200A overhead

APPLICABLE CODES (Wisconsin 2025-2026):
- Building: 2021 IBC (eff. Oct 1, 2025)
- Electrical: 2023 NEC (eff. Jan 1, 2025) — E-001 on drawings incorrectly references 2017 NEC + ASHRAE 90.1-2013; Seifert must reverify all work
- Energy: 2021 IECC / Wisconsin SPS
- Fire Sprinkler: NFPA 13 (2025 ed.)
- Fire Alarm: NFPA 72 (2025 ed.) — NOT required (occupancy < 500 persons)
- Mechanical: 2021 IMC / NFPA 90B 2024
- Plumbing: Wisconsin SPS 381-387
- GFCI DEADLINE: NEC 210.8(F) — GFCI at all outdoor HVAC equipment by September 1, 2026

DOCUMENTS: Original set (39 pages, 10/30/2020) + REV #4 (10 sheets + narrative, KLH Engineers, 2/25/2026)
ASBESTOS: 1,500 SF black mastic ACM confirmed (Vertex #102403, Tucker Ryckman WI Lic AII-253634) — landlord remediating

══════════════════════════════════════════════════════════════
RTU / HVAC INVENTORY (AUTHORITATIVE — S2 field-confirmed)
══════════════════════════════════════════════════════════════
TERMINOLOGY: RTU = HVAC = same unit. "X" prefix (XRTU) = existing unit to remain. HVAC-1 thru HVAC-5 on EMS drawings = RTU-1 thru RTU-5 (1:1).

RTU-1  | 1,250 LBS | REPLACED — new unit | Existing location (stays put) | W24 STL. BEAM + DBL. JOIST | Owner-furnished, Central Temp installs
RTU-2  | 1,250 LBS | REPLACED — new unit | ★ NEW LOCATION (moved)        | W27 STL. BEAM + DBL. JOIST | Owner-furnished; new roof penetration; 4 trades affected
XRTU-3 | —         | STAYS — service only | Exist. to remain              | W27 STL. BEAM               | Existing unit; NOT replaced; Δ2 cloud on S2
RTU-4  | 755 LBS   | REPLACED — new unit | Existing location (stays put) | W24 STL. BEAM + DBL. JOIST | Owner-furnished; Δ1 cloud on S2
XRTU-5 | —         | STAYS — service only | Exist. to remain (bottom)     | —                           | Existing unit; NOT replaced

3 owner-furnished replacements (Central Temp installs): RTU-1, RTU-2, RTU-4
2 existing-to-remain (service/commission only): XRTU-3, XRTU-5
RTU-2 RELOCATION impacts: Home Insulation (patch old curb/cut new penetration) | Best1 (re-route gas branch) | Seifert (re-route circuit + new disconnect) | Central Temp (new ducts, condensate trap at new curb)

══════════════════════════════════════════════════════════════
EMS SYSTEM (CRITICAL — READ CAREFULLY)
══════════════════════════════════════════════════════════════
CORRECT SYSTEM: CYLON by Schneider Electric (EcoStruxure platform)
WARNING: EM-102 and EM-103 drawings still show SimpleStat/eSCi — these drawings were NOT updated in REV #4.
Seifert Electric must NOT wire from EM-102/EM-103 until KLH issues revised EMS drawings (RFI #53 open).
HVAC-1 thru HVAC-5 on EM drawings = RTU-1 thru RTU-5 (1:1 confirmed — RFI #54 closed).
CO2 sensors and Remote Space Temperature Sensors (STS on downrods) are still required.

══════════════════════════════════════════════════════════════
OPEN CRITICAL RFIs (as of 02/27/2026)
══════════════════════════════════════════════════════════════
#39  OPEN | Economizer required on ALL new RTUs — IECC C403.5 / SPS 363.0403(3): economizer + fault detection + Class I motorized damper on RTU-1, RTU-2, RTU-4. No confirmation from Dollar Tree yet.
#51  OPEN | XRTU-3 AND XRTU-5 service scope — Central Temp NTP covers 3 replacements only ($36,778). Service of both existing units = 2 potential change orders. Call Chad DuFrane 920-731-5071.
#52  OPEN | Seifert re-verify RTU-1/RTU-2 circuits vs. E-102/E-202 REV #4 delta clouds.
#53  OPEN | EM-102/EM-103 not updated for CYLON EMS system — KLH must issue revised drawings before Seifert begins any EMS wiring.
#54  CLOSED | HVAC-1 thru HVAC-5 = RTU-1 thru RTU-5 numbering confirmed 1:1.
#55  OPEN | RTU-4 new unit weight unknown — if heavier than 755 LBS, W24 beam may be undersized. Need equipment submittal from Dollar Tree.
#56  OPEN | Economizer/damper confirmation on owner-furnished RTU-1, RTU-2, RTU-4 — get submittals from Dollar Tree before units ship.
#57  OPEN | Condensate drain relocation at RTU-2 new position — no sheet assigns this work. Coordinate Central Temp + Best1 + Home Insulation.
#58  OPEN | Laser Fire hazard classification for DSPS hydraulic calcs — not stated on FP2 drawings. Sales floor = Light Hazard; Stockroom = Ordinary Hazard Group 1 (likely). Confirm before DSPS submittal.

══════════════════════════════════════════════════════════════
SUBCONTRACTOR CONTACTS & NTP AMOUNTS
══════════════════════════════════════════════════════════════
Laser Fire Protection    | David Bartolerio | 608-205-7219 | WI Lic #948334   | $17,100  | Net 30
Home Insulation (Roof)   | Zachary Chaignot | 715-359-6505 | GC expense work   | $2,935   | Net 30
Seifert Electric         | Kim Kluz         | 715-693-2625 |                   | $45,450  | Net 30
Best1 Plumbing           | Zach/Nate Roth   | 715-241-0883 |                   | $25,415  | Net 30
Central Temp (HVAC)      | Chad DuFrane     | 920-731-5071 |                   | $36,778  | ⚠ NET 10 DAYS (pay promptly)

PROJECT TEAM:
GC Superintendent: T.C. DiYanni (Sun Industries)
CM / Owner Rep: Will Sparks (Sun PM) | 281-706-5894
DT Project Mgr: Miguel Sanchez Leos (Dollar Tree)
CCI: Kyle Spangler | 314-991-2633
Architect: Brian Eady, M Architects | 586-933-3010 | brian@marchitects.com | WI Lic A-15489-5
MEP Engineer: KLH Engineers (KLH Project #27191.00) | Ben Justice PM | 859-303-3715
Landlord: Ned Brickman, RCM Wausau LLC / Midland Mgt | 414-852-1074 | nbrickman@midlandmgtllc.com
Structural: Broyles and Associates

INSPECTORS / AUTHORITIES:
DSPS Plan Reviewer: Jason Hansen | 920-492-7728 | jason.hansen@wisconsin.gov
DIS Inspector: Jon Molledahl | 608-225-6520 | jon.molledahl@wisconsin.gov
Municipal Clerk (Rothschild): Elizabeth Felkner | 715-359-3660 | efelkner@rothschildwi.com
DSPS Fire Protech: DSPSSBFireprotech@wisconsin.gov
Supervising Architect: Branin Gries (Gries Design) | 920-722-2445
Supervising Engineer: Jodi Flaherty P.E. (CMG & Assoc.) | 608-842-0700 | jflaherty@cmgengineers.com

══════════════════════════════════════════════════════════════
LIVE PROJECT DATA (from database — updated in real time)
══════════════════════════════════════════════════════════════
Project: ${project?.name || 'Unknown'} | ${project?.location || 'Unknown'}
Trade Focus: ${trade.toUpperCase()} | Codes: ${codes}

LEARNED INSIGHTS (AI learning loop — ${insights.length} loops completed):
${insightSummary}

Documents on file (${docs.length} total):
${docSummary || "No documents uploaded yet."}

Open RFIs (live — ${openRfis.length} open of ${rfis.length} total):
${rfiSummary || "No open RFIs."}

Recent Daily Logs (last 5):
${logSummary || "No logs yet."}

══════════════════════════════════════════════════════════════
PUNCHLIST STATUS (live)
══════════════════════════════════════════════════════════════
${punchSummary || "No punchlist items yet."}

══════════════════════════════════════════════════════════════
FIELD EVENTS LOG (last 8)
══════════════════════════════════════════════════════════════
${recentEvents || "No field events logged yet."}

══════════════════════════════════════════════════════════════
SUBMITTALS STATUS (pending only)
══════════════════════════════════════════════════════════════
${submittalSummary || "No pending submittals."}

══════════════════════════════════════════════════════════════
CONTRACTOR STATUS (live from DB)
══════════════════════════════════════════════════════════════
${contractorSummary || "No contractors in DB — use authoritative contacts above."}

══════════════════════════════════════════════════════════════
INSTRUCTIONS
══════════════════════════════════════════════════════════════
- AUTHORITATIVE KNOWLEDGE above is always correct — trust it over DB entries for hardcoded facts
- Use learned insights as refined project-specific intelligence built up over time
- Always cite specific code sections (NEC, NFPA, IMC, IECC, Wisconsin SPS) when answering code questions
- Cross-reference punchlist, field events, and open RFIs proactively when relevant
- Flag payment terms (especially NET 10 DAYS for Central Temp) when payment comes up
- Give field-ready, actionable answers — this is a mobile tool used on a jobsite
- When contacts are needed, give the name AND phone number
- Warn immediately if anything the user describes conflicts with an open RFI, punchlist critical item, or field event`;

      const recentHistory = history.slice(-20);

      // Build Claude-compatible message array (alternating user/assistant roles)
      // Claude requires the conversation to start with a user turn, so we
      // filter out any leading assistant messages.
      const claudeMessages: Anthropic.MessageParam[] = recentHistory
        .filter((_, idx, arr) => {
          // Drop leading assistant messages — Claude requires user turn first
          const firstUserIdx = arr.findIndex(m => m.role === "user");
          return idx >= firstUserIdx;
        })
        .map(c => ({
          role: c.role === "assistant" ? "assistant" : "user",
          content: c.content,
        }));

      // Ensure conversation ends with a user message (append if needed)
      if (claudeMessages.length === 0 || claudeMessages[claudeMessages.length - 1].role !== "user") {
        claudeMessages.push({ role: "user", content: message });
      }

      const claude = getClaudeClient();
      const claudeResponse = await claude.messages.create({
        model: "claude-sonnet-4-5",
        max_tokens: 4096,
        system: systemMessage,
        messages: claudeMessages,
      });

      const assistantText =
        claudeResponse.content
          .filter((b): b is Anthropic.TextBlock => b.type === "text")
          .map(b => b.text)
          .join("") || "I couldn't process that request.";
      const newChat = await storage.createChat(projectId, { role: "assistant", content: assistantText });
      res.status(201).json(newChat);
    } catch (err) {
      console.error(err);
      if (err instanceof z.ZodError)
        return res.status(400).json({ message: err.errors[0].message, field: err.errors[0].path.join('.') });
      res.status(500).json({ message: "Internal error processing chat." });
    }
  });

  // ── Export: RFI Sheet ──────────────────────────────────────────────────────
  app.get("/api/projects/:id/export/rfis", async (req, res) => {
    const projectId = Number(req.params.id);
    const [project, rfis] = await Promise.all([
      storage.getProject(projectId),
      storage.getRfis(projectId),
    ]);
    if (!project) return res.status(404).json({ message: "Project not found" });

    const now = new Date().toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" });
    const byTrade: Record<string, typeof rfis> = {};
    for (const rfi of rfis) {
      if (!byTrade[rfi.trade]) byTrade[rfi.trade] = [];
      byTrade[rfi.trade].push(rfi);
    }

    let out = `RFI REPORT — ${project.name.toUpperCase()}\n`;
    out += `${project.location}\n`;
    out += `Generated: ${now}\n`;
    out += `Total RFIs: ${rfis.length}\n`;
    out += `${"═".repeat(70)}\n\n`;

    const tradeOrder = ["general", "hvac", "electrical", "plumbing", "fire"];
    for (const trade of [...tradeOrder, ...Object.keys(byTrade).filter(t => !tradeOrder.includes(t))]) {
      const list = byTrade[trade];
      if (!list?.length) continue;
      const label = trade === "hvac" ? "HVAC" : trade.toUpperCase();
      out += `${"─".repeat(70)}\n`;
      out += `TRADE: ${label} (${list.length} RFI${list.length > 1 ? "s" : ""})\n`;
      out += `${"─".repeat(70)}\n`;
      list.forEach((rfi, i) => {
        out += `\n[${i + 1}] ${rfi.title}\n`;
        out += `    Status: ${rfi.status.toUpperCase()}\n`;
        out += `    ${rfi.description?.replace(/\n/g, "\n    ")}\n`;
      });
      out += "\n";
    }

    out += `${"═".repeat(70)}\n`;
    out += `END OF RFI REPORT\n`;

    res.setHeader("Content-Type", "text/plain; charset=utf-8");
    res.setHeader("Content-Disposition", `attachment; filename="${project.name.replace(/[^a-z0-9]/gi, "_")}_RFIs.txt"`);
    res.send(out);
  });

  // ── Export: Scope of Work by Trade ────────────────────────────────────────
  app.get("/api/projects/:id/export/scope/:trade", async (req, res) => {
    const projectId = Number(req.params.id);
    const trade = req.params.trade;
    const [project, docs, rfis] = await Promise.all([
      storage.getProject(projectId),
      storage.getDocuments(projectId),
      storage.getRfis(projectId),
    ]);
    if (!project) return res.status(404).json({ message: "Project not found" });

    const now = new Date().toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" });
    const label = trade === "hvac" ? "HVAC" : trade.charAt(0).toUpperCase() + trade.slice(1);
    const tradeRfis = rfis.filter(r => r.trade === trade || r.trade === "general");
    const tradeDocs = docs.filter(d => d.trade === trade || d.trade === "general");
    const codes = TRADE_CODES[trade] || TRADE_CODES.general;

    let out = `SCOPE OF WORK — ${label.toUpperCase()} TRADE\n`;
    out += `${project.name} | ${project.location}\n`;
    out += `Generated: ${now}\n`;
    out += `${"═".repeat(70)}\n\n`;

    out += `APPLICABLE CODES\n${"─".repeat(70)}\n${codes}\n\n`;

    if (tradeDocs.length) {
      out += `${"─".repeat(70)}\nDOCUMENTS (${tradeDocs.length})\n${"─".repeat(70)}\n`;
      tradeDocs.forEach(d => {
        out += `\n• ${d.name} [${d.type}]\n`;
        if (d.content && !d.content.startsWith("Blueprint uploaded") && !d.content.startsWith("File uploaded")) {
          out += `  ${d.content.replace(/\n/g, "\n  ")}\n`;
        }
      });
      out += "\n";
    }

    if (tradeRfis.length) {
      out += `${"─".repeat(70)}\nRFIs (${tradeRfis.length})\n${"─".repeat(70)}\n`;
      tradeRfis.forEach((rfi, i) => {
        out += `\n[${i + 1}] [${rfi.status.toUpperCase()}] ${rfi.title}\n`;
        out += `  ${rfi.description?.replace(/\n/g, "\n  ")}\n`;
      });
    }

    out += `\n${"═".repeat(70)}\nEND OF ${label.toUpperCase()} SCOPE\n`;

    res.setHeader("Content-Type", "text/plain; charset=utf-8");
    res.setHeader("Content-Disposition", `attachment; filename="${project.name.replace(/[^a-z0-9]/gi, "_")}_${label}_Scope.txt"`);
    res.send(out);
  });

  // ── Export: All Scopes Combined ────────────────────────────────────────────
  app.get("/api/projects/:id/export/scope", async (req, res) => {
    const projectId = Number(req.params.id);
    const [project, docs, rfis] = await Promise.all([
      storage.getProject(projectId),
      storage.getDocuments(projectId),
      storage.getRfis(projectId),
    ]);
    if (!project) return res.status(404).json({ message: "Project not found" });

    const now = new Date().toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" });
    let out = `COMPLETE SCOPE OF WORK — ALL TRADES\n`;
    out += `${project.name} | ${project.location}\n`;
    out += `Generated: ${now}\n`;
    out += `${"═".repeat(70)}\n\n`;
    out += `PROJECT DESCRIPTION\n${"─".repeat(70)}\n${project.description}\n\n`;

    const trades = ["general", "hvac", "electrical", "plumbing", "fire"];
    for (const trade of trades) {
      const label = trade === "hvac" ? "HVAC" : trade.charAt(0).toUpperCase() + trade.slice(1);
      const codes = TRADE_CODES[trade] || TRADE_CODES.general;
      const tradeRfis = rfis.filter(r => r.trade === trade);
      const tradeDocs = docs.filter(d => d.trade === trade);
      out += `${"═".repeat(70)}\n${label.toUpperCase()} TRADE\n${"═".repeat(70)}\n`;
      out += `\nAPPLICABLE CODES:\n${codes}\n`;
      if (tradeDocs.length) {
        out += `\nDOCUMENTS:\n`;
        tradeDocs.forEach(d => {
          out += `• ${d.name} [${d.type}]\n`;
          if (d.content && !d.content.startsWith("Blueprint") && !d.content.startsWith("File")) {
            out += `  ${d.content.slice(0, 600).replace(/\n/g, "\n  ")}\n`;
          }
        });
      }
      if (tradeRfis.length) {
        out += `\nOPEN RFIs (${tradeRfis.length}):\n`;
        tradeRfis.forEach(r => { out += `• [${r.status.toUpperCase()}] ${r.title}\n`; });
      }
      out += "\n";
    }
    out += `${"═".repeat(70)}\nEND OF COMPLETE SCOPE\n`;

    res.setHeader("Content-Type", "text/plain; charset=utf-8");
    res.setHeader("Content-Disposition", `attachment; filename="${project.name.replace(/[^a-z0-9]/gi, "_")}_Complete_Scope.txt"`);
    res.send(out);
  });


  // Trade Contractors
  app.get('/api/projects/:id/contractors', async (req, res) => {
    try {
      const contractors = await storage.getTradeContractors(Number(req.params.id));
      res.json(contractors);
    } catch (err) {
      res.status(500).json({ message: 'Failed to fetch contractors' });
    }
  });

  app.post('/api/projects/:id/contractors/:trade', async (req, res) => {
    try {
      const { companyName, contactName, contactPhone, contactEmail, notes } = req.body;
      if (!companyName) return res.status(400).json({ message: 'companyName is required' });
      const contractor = await storage.upsertTradeContractor(
        Number(req.params.id),
        req.params.trade,
        { companyName, contactName, contactPhone, contactEmail, notes }
      );
      res.status(201).json(contractor);
    } catch (err) {
      res.status(500).json({ message: 'Failed to save contractor' });
    }
  });

  app.delete('/api/projects/:id/contractors/:contractorId', async (req, res) => {
    try {
      await storage.deleteTradeContractor(Number(req.params.contractorId));
      res.json({ success: true });
    } catch (err) {
      res.status(500).json({ message: 'Failed to delete contractor' });
    }
  });

  // Full Scope AI Analysis per Trade
  app.post('/api/projects/:id/scope/:trade/run', async (req, res) => {
    try {
      const projectId = Number(req.params.id);
      const trade = req.params.trade;
      const [project, docs, rfisData, logsData, contractors] = await Promise.all([
        storage.getProject(projectId),
        storage.getDocuments(projectId),
        storage.getRfis(projectId),
        storage.getDailyLogs(projectId),
        storage.getTradeContractors(projectId),
      ]);
      if (!project) return res.status(404).json({ message: 'Project not found' });

      const label = trade === 'hvac' ? 'HVAC' : trade.charAt(0).toUpperCase() + trade.slice(1);
      const codes = TRADE_CODES[trade] || TRADE_CODES.general;
      const contractor = contractors.find(c => c.trade === trade);
      const tradeDocs = docs.filter(d => d.trade === trade || d.trade === 'general');
      const tradeRfis = rfisData.filter(r => r.trade === trade || r.trade === 'general');
      const tradeLogs = logsData.filter(l => l.trade === trade);

      let contractorInfo = 'No contractor assigned yet.';
      if (contractor) {
        contractorInfo = 'Assigned Contractor: ' + contractor.companyName;
        if (contractor.contactName) contractorInfo += ' | Contact: ' + contractor.contactName;
        if (contractor.contactPhone) contractorInfo += ' | Phone: ' + contractor.contactPhone;
        if (contractor.contactEmail) contractorInfo += ' | Email: ' + contractor.contactEmail;
        if (contractor.notes) contractorInfo += ' | Notes: ' + contractor.notes;
      }

      const docSummary = tradeDocs.map(d => '- ' + d.name + ' [' + d.type + ']: ' + (d.content?.slice(0, 400) || 'no content')).join('\n');
      const rfiSummary = tradeRfis.map(r => '- [' + r.status.toUpperCase() + '] ' + r.title + ': ' + (r.description?.slice(0, 200) || '')).join('\n');
      const logSummary = tradeLogs.slice(-10).map(l => '- [' + l.date + '] ' + l.completedWork + (l.discrepancies ? ' | Issues: ' + l.discrepancies : '')).join('\n');

      const userPrompt = [
        'Generate a comprehensive FULL SCOPE REPORT for the ' + label + ' trade on this project.',
        '',
        'Project: ' + project.name,
        'Location: ' + project.location,
        'Trade: ' + label,
        '',
        contractorInfo,
        '',
        'APPLICABLE CODES:',
        codes,
        '',
        'DOCUMENTS ON FILE (' + tradeDocs.length + '):',
        docSummary || 'None',
        '',
        'OPEN RFIs (' + tradeRfis.length + '):',
        rfiSummary || 'None',
        '',
        'RECENT FIELD LOGS (' + tradeLogs.length + '):',
        logSummary || 'None',
        '',
        'Structure your report with these sections:',
        '1. SCOPE SUMMARY',
        '2. CONTRACTOR ASSIGNMENT',
        '3. APPLICABLE CODES & COMPLIANCE',
        '4. DETAILED SCOPE OF WORK',
        '5. SUBMITTALS & DOCUMENTATION REQUIRED',
        '6. COORDINATION REQUIREMENTS',
        '7. OPEN ISSUES & RFIs',
        '8. INSPECTION & CLOSEOUT CHECKLIST',
        '9. RISK FLAGS',
        '10. RECOMMENDED NEXT ACTIONS',
      ].join('\n');

      const gemini = getGeminiClient();
      const response = await gemini.models.generateContent({
        model: 'gemini-2.5-flash',
        contents: [{ role: 'user', parts: [{ text: userPrompt }] }],
        config: {
          systemInstruction: 'You are an expert construction project manager and code compliance specialist. Provide detailed, actionable, field-ready scope reports.',
          maxOutputTokens: 8192,
        },
      });

      const scopeText = response.text || 'No scope generated.';
      res.json({ trade, label, scope: scopeText, contractorInfo });
    } catch (error) {
      console.error('Scope run error:', error);
      res.status(500).json({ message: 'Failed to run scope analysis. Please try again.' });
    }
  });


  // Learning loop status
  app.get("/api/projects/:id/learning/status", async (req, res) => {
    try {
      const status = await getLoopStatus(Number(req.params.id));
      res.json(status);
    } catch (err: any) {
      res.status(500).json({ message: err.message });
    }
  });

  app.get("/api/projects/:id/learning/insights", async (req, res) => {
    try {
      const insights = await storage.getInsights(Number(req.params.id), 10);
      res.json(insights);
    } catch (err: any) {
      res.status(500).json({ message: err.message });
    }
  });

  // ── Field Events ─────────────────────────────────────────────────────────
  app.get("/api/projects/:id/field-events", async (req, res) => {
    try {
      const events = await storage.getFieldEvents(Number(req.params.id));
      res.json(events);
    } catch (err: any) {
      res.status(500).json({ message: err.message });
    }
  });

  app.post("/api/projects/:id/field-events", async (req, res) => {
    try {
      const projectId = Number(req.params.id);
      const body = req.body;
      if (!body.type || !body.description || !body.date) {
        return res.status(400).json({ message: "type, description, and date are required" });
      }
      const event = await storage.createFieldEvent(projectId, body);
      res.status(201).json(event);
    } catch (err: any) {
      res.status(500).json({ message: err.message });
    }
  });

  app.patch("/api/field-events/:id", async (req, res) => {
    try {
      const event = await storage.updateFieldEvent(Number(req.params.id), req.body);
      res.json(event);
    } catch (err: any) {
      res.status(500).json({ message: err.message });
    }
  });

  app.delete("/api/field-events/:id", async (req, res) => {
    try {
      await storage.deleteFieldEvent(Number(req.params.id));
      res.json({ success: true });
    } catch (err: any) {
      res.status(500).json({ message: err.message });
    }
  });

  // Promote a field event into a lesson
  app.post("/api/field-events/:id/promote", async (req, res) => {
    try {
      const event = await storage.getFieldEvent(Number(req.params.id));
      if (!event) return res.status(404).json({ message: "Event not found" });

      const lesson = await storage.createLesson({
        projectId: event.projectId,
        eventId: event.id,
        title: req.body.title || `Lesson from ${event.type} — ${event.date}`,
        projectContext: req.body.projectContext || null,
        problem: req.body.problem || event.description,
        rootCause: req.body.rootCause || null,
        fix: req.body.fix || null,
        prevention: req.body.prevention || null,
        trade: event.trade || null,
        tags: req.body.tags || null,
      });

      await storage.updateFieldEvent(event.id, { status: "promoted" });
      res.status(201).json(lesson);
      embedLessonAsync(lesson).catch(() => {});
    } catch (err: any) {
      res.status(500).json({ message: err.message });
    }
  });

  // ── Lessons Learned ───────────────────────────────────────────────────────
  app.get("/api/lessons", async (req, res) => {
    try {
      const projectId = req.query.projectId ? Number(req.query.projectId) : undefined;
      const q = req.query.q as string | undefined;
      let result;
      if (q && q.trim()) {
        result = await storage.searchLessons(q.trim());
        if (projectId !== undefined) result = result.filter(l => l.projectId === projectId);
      } else {
        result = await storage.getLessons(projectId);
      }
      res.json(result);
    } catch (err: any) {
      res.status(500).json({ message: err.message });
    }
  });

  app.post("/api/lessons", async (req, res) => {
    try {
      const body = req.body;
      if (!body.projectId || !body.title || !body.problem) {
        return res.status(400).json({ message: "projectId, title, and problem are required" });
      }
      const lesson = await storage.createLesson(body);
      res.status(201).json(lesson);
      embedLessonAsync(lesson).catch(() => {});
    } catch (err: any) {
      res.status(500).json({ message: err.message });
    }
  });

  app.patch("/api/lessons/:id", async (req, res) => {
    try {
      const lesson = await storage.updateLesson(Number(req.params.id), req.body);
      res.json(lesson);
      embedLessonAsync(lesson).catch(() => {});
    } catch (err: any) {
      res.status(500).json({ message: err.message });
    }
  });

  app.delete("/api/lessons/:id", async (req, res) => {
    try {
      await storage.deleteLesson(Number(req.params.id));
      res.json({ success: true });
    } catch (err: any) {
      res.status(500).json({ message: err.message });
    }
  });

  // ── AI Recommendations (vector-powered) ──────────────────────────────────

  async function buildRecommendation(situation: string, trade: string | undefined, projectType: string | undefined, phase: string | undefined): Promise<string> {
    const gemini = getGeminiClient();

    // 1. Try semantic (vector) search first
    let topLessons: any[] = [];
    const queryEmbedding = await generateEmbedding(situation);
    if (queryEmbedding) {
      topLessons = await storage.vectorSearchLessons(queryEmbedding, 12);
    }

    // 2. Fall back / supplement with full library (for context + non-embedded lessons)
    const allLessons = await storage.getLessons();
    const topIds = new Set(topLessons.map((l: any) => l.id));
    const remainingLessons = allLessons.filter(l => !topIds.has(l.id)).slice(0, 20);

    const formatLesson = (l: any) =>
      `[#${l.id}] ${l.title}\nTrade: ${l.trade || "General"} | Tags: ${(l.tags || []).join(", ") || "none"}\nProblem: ${l.problem}\nRoot Cause: ${l.rootCause || "N/A"}\nFix: ${l.fix || "N/A"}\nPrevention: ${l.prevention || "N/A"}`;

    const semanticSection = topLessons.length > 0
      ? `SEMANTICALLY MATCHED LESSONS (most relevant first):\n${topLessons.map(formatLesson).join("\n\n")}`
      : "";
    const remainingSection = remainingLessons.length > 0
      ? `ADDITIONAL LIBRARY LESSONS:\n${remainingLessons.map(formatLesson).join("\n\n")}`
      : "";
    const librarySection = [semanticSection, remainingSection].filter(Boolean).join("\n\n---\n\n") || "No lessons in library yet.";

    const prompt = `You are a senior construction superintendent AI. A superintendent is facing this situation:

SITUATION: ${situation}
TRADE: ${trade || "Not specified"}
PROJECT TYPE: ${projectType || "Commercial construction"}
PHASE: ${phase || "Not specified"}

${librarySection}

Based on the situation and the matched lessons above, provide:
1. WATCH-OUTS: Key risks and things to watch for (bullet list)
2. RECOMMENDED SEQUENCE: Suggested steps or sequence to handle this situation
3. RELEVANT LESSONS: Reference specific lessons from the library (by title) that apply
4. CHECKLIST: A quick pre-start checklist specific to this situation (8-12 items)

Be specific, practical, and concise. Focus on field-level guidance a superintendent can act on today.`;

    const completion = await gemini.models.generateContent({
      model: "gemini-2.5-flash",
      contents: [{ role: "user", parts: [{ text: prompt }] }],
      config: { maxOutputTokens: 1400 },
    });
    return completion.text || "";
  }

  // POST /api/lessons/recommend (original, kept for backward compat)
  app.post("/api/lessons/recommend", async (req, res) => {
    try {
      const { situation, trade, projectType, phase } = req.body;
      if (!situation) return res.status(400).json({ message: "situation is required" });
      const recommendation = await buildRecommendation(situation, trade, projectType, phase);
      res.json({ recommendation });
    } catch (err: any) {
      res.status(500).json({ message: err.message });
    }
  });

  // GET /api/recommendations?situation=...&trade=...&project_type=...&phase=...
  app.get("/api/recommendations", async (req, res) => {
    try {
      const situation = req.query.situation as string | undefined;
      if (!situation) return res.status(400).json({ message: "situation query param is required" });
      const trade = req.query.trade as string | undefined;
      const projectType = req.query.project_type as string | undefined;
      const phase = req.query.phase as string | undefined;
      const recommendation = await buildRecommendation(situation, trade, projectType, phase);
      res.json({ recommendation, situation, trade, projectType, phase });
    } catch (err: any) {
      res.status(500).json({ message: err.message });
    }
  });

  // POST /api/lessons/embed-all — backfill embeddings for all un-embedded lessons
  app.post("/api/lessons/embed-all", async (req, res) => {
    try {
      const pending = await storage.getLessonsWithoutEmbedding();
      res.json({ message: `Backfilling ${pending.length} lessons in background` });
      for (const lesson of pending) {
        await embedLessonAsync(lesson);
      }
    } catch (err: any) {
      res.status(500).json({ message: err.message });
    }
  });

  // POST /api/projects/:id/documents/analyze-all — backfill analysis for any unanalyzed uploaded files
  app.post("/api/projects/:id/documents/analyze-all", async (req, res) => {
    try {
      const projectId = Number(req.params.id);
      const docs = await storage.getDocuments(projectId);
      const unanalyzed = docs.filter(d =>
        !d.content ||
        d.content.startsWith("File uploaded") ||
        d.content.startsWith("Blueprint uploaded") ||
        d.content === "Analyzing document…"
      );
      res.json({ message: `Backfilling analysis for ${unanalyzed.length} document(s) in background` });
      for (const doc of unanalyzed) {
        await autoAnalyzeDocument(doc.id, projectId);
      }
    } catch (err: any) {
      res.status(500).json({ message: err.message });
    }
  });

  // ─── Blueprint Intelligence ────────────────────────────────────────────────
  app.get("/api/projects/:id/blueprint-analyses", async (req, res) => {
    try {
      const analyses = await storage.getBlueprintAnalyses(parseInt(req.params.id));
      res.json(analyses);
    } catch (err: any) { res.status(500).json({ message: err.message }); }
  });

  app.get("/api/blueprint-analyses/:id", async (req, res) => {
    try {
      const analysis = await storage.getBlueprintAnalysis(parseInt(req.params.id));
      if (!analysis) return res.status(404).json({ message: "Not found" });
      res.json(analysis);
    } catch (err: any) { res.status(500).json({ message: err.message }); }
  });

  app.delete("/api/blueprint-analyses/:id", async (req, res) => {
    try {
      await storage.deleteBlueprintAnalysis(parseInt(req.params.id));
      res.json({ success: true });
    } catch (err: any) { res.status(500).json({ message: err.message }); }
  });

  app.post("/api/projects/:id/blueprint-analyses", async (req, res) => {
    try {
      const projectId = parseInt(req.params.id);
      const { documentIds, analysisType = "full", name } = req.body as { documentIds: number[], analysisType: string, name: string };
      if (!documentIds || !documentIds.length) return res.status(400).json({ message: "documentIds required" });

      const project = await storage.getProject(projectId);
      const location = project?.location || "Unknown";

      const pending = await storage.createBlueprintAnalysis({
        projectId,
        name: name || `Analysis ${new Date().toLocaleDateString()}`,
        analysisType,
        documentIds: documentIds.join(","),
        status: "running",
      });

      res.json(pending);

      // Run analysis async
      (async () => {
        try {
          const docs = await Promise.all(documentIds.map((id: number) => storage.getDocument(id)));
          const validDocs = docs.filter(d => d && d.fileData && d.fileMime?.startsWith("image/"));

          if (!validDocs.length) {
            await storage.updateBlueprintAnalysis(pending.id, {
              status: "error",
              summary: "No image documents found. Upload PNG/JPG blueprints and try again.",
            });
            return;
          }

          const ANALYSIS_TYPES: Record<string, string> = {
            "clash": "Focus entirely on MEP coordination clashes: pipes through beams, duct conflicts, equipment access conflicts. Be exhaustive — check every potential intersection.",
            "elements": "Focus on extracting all visible elements: equipment tags, fixture types, panel labels, duct sizes, pipe sizes, dimensions. Create a complete inventory.",
            "code": "Focus on code compliance: check NEC 2023, 2021 IBC, NFPA 72/13, Wisconsin SPS codes. Flag every potential violation with code section.",
            "full": "Perform a complete analysis: clash detection, element extraction, code compliance, and missing element identification."
          };
          const focusInstruction = ANALYSIS_TYPES[analysisType] || ANALYSIS_TYPES.full;

          const systemPrompt = `You are an expert MEP coordination engineer and licensed construction superintendent analyzing architectural and engineering drawings for a commercial construction project.

Project: ${project?.name || "Unknown"} — ${location}
Applicable codes: 2021 IBC, 2023 NEC, NFPA 72 (2025), NFPA 13, Wisconsin SPS 361/362/363/381/385

${focusInstruction}

You are analyzing ${validDocs.length} drawing(s) together as a cross-trade coordination review.

IMPORTANT: Return ONLY valid JSON in exactly this format, no markdown, no explanation:
{
  "summary": "2-3 sentence executive summary of findings",
  "clashes": [
    { "severity": "high|medium|low", "trades": ["HVAC", "Structural"], "location": "Grid B-4 / Ceiling plenum", "description": "24x12 supply duct conflicts with W12x35 beam — approximately 6 inches of clearance lost", "recommendation": "Reroute duct 18 inches east or lower beam soffit" }
  ],
  "elements": [
    { "trade": "HVAC|Electrical|Plumbing|Structural|Fire|General", "type": "RTU|Panel|Fixture|Beam|Sprinkler|etc", "tag": "RTU-1", "location": "Roof / Grid A-B/1-3", "size": "5 ton / 2000 CFM", "notes": "" }
  ],
  "codeIssues": [
    { "severity": "high|medium|low", "code": "NEC 110.26", "description": "Electrical panel clearance appears insufficient — minimum 36 inches required", "location": "Electrical room", "recommendation": "Verify 36\" working clearance in front of all panel boards" }
  ],
  "missingItems": [
    { "item": "Emergency disconnect for HVAC equipment", "required_by": "NEC 440.14", "location": "RTU-1 location", "action": "Add within sight of unit or provide lockable disconnect" }
  ]
}`;

          const imageMessages = validDocs.map((d: any) => ({
            type: "image_url" as const,
            image_url: { url: `data:${d!.fileMime};base64,${d!.fileData}`, detail: "high" as const }
          }));

          const gpt = getGptClient();
          const response = await gpt.chat.completions.create({
            model: "gpt-4o",
            max_tokens: 8192,
            messages: [
              { role: "system", content: systemPrompt },
              {
                role: "user",
                content: [
                  ...imageMessages,
                  { type: "text", text: `Analyze ${validDocs.map((d: any) => d!.name).join(", ")} for ${analysisType === "full" ? "complete multi-trade coordination review" : analysisType + " analysis"}. Return only the JSON structure.` }
                ]
              }
            ],
          });

          const rawText = response.choices[0]?.message?.content || "{}";
          let parsed: any = {};
          try {
            const jsonMatch = rawText.match(/\{[\s\S]*\}/);
            parsed = jsonMatch ? JSON.parse(jsonMatch[0]) : {};
          } catch {
            parsed = { summary: rawText };
          }

          await storage.updateBlueprintAnalysis(pending.id, {
            status: "complete",
            clashes: parsed.clashes || [],
            elements: parsed.elements || [],
            codeIssues: parsed.codeIssues || [],
            missingItems: parsed.missingItems || [],
            summary: parsed.summary || "Analysis complete.",
            rawResponse: rawText,
          });
        } catch (err: any) {
          await storage.updateBlueprintAnalysis(pending.id, {
            status: "error",
            summary: `Analysis failed: ${err.message}`,
          });
        }
      })();
    } catch (err: any) { res.status(500).json({ message: err.message }); }
  });

  // ─── RFI enhanced CRUD ─────────────────────────────────────────────────────
  app.patch("/api/rfis/:id", async (req, res) => {
    try {
      const id = parseInt(req.params.id);
      const updated = await storage.updateRfi(id, req.body);
      res.json(updated);
    } catch (err: any) { res.status(500).json({ message: err.message }); }
  });
  app.delete("/api/rfis/:id", async (req, res) => {
    try {
      await storage.deleteRfi(parseInt(req.params.id));
      res.json({ success: true });
    } catch (err: any) { res.status(500).json({ message: err.message }); }
  });

  // ─── HVAC Equipment ────────────────────────────────────────────────────────
  app.get("/api/projects/:id/hvac", async (req, res) => {
    try {
      const items = await storage.getHvacEquipment(parseInt(req.params.id));
      res.json(items);
    } catch (err: any) { res.status(500).json({ message: err.message }); }
  });
  app.post("/api/projects/:id/hvac", async (req, res) => {
    try {
      const item = await storage.createHvacEquipment(parseInt(req.params.id), req.body);
      res.json(item);
    } catch (err: any) { res.status(500).json({ message: err.message }); }
  });
  app.patch("/api/hvac/:id", async (req, res) => {
    try {
      const item = await storage.updateHvacEquipment(parseInt(req.params.id), req.body);
      res.json(item);
    } catch (err: any) { res.status(500).json({ message: err.message }); }
  });
  app.delete("/api/hvac/:id", async (req, res) => {
    try {
      await storage.deleteHvacEquipment(parseInt(req.params.id));
      res.json({ success: true });
    } catch (err: any) { res.status(500).json({ message: err.message }); }
  });

  // ─── Electrical Panels ─────────────────────────────────────────────────────
  app.get("/api/projects/:id/electrical/panels", async (req, res) => {
    try {
      const items = await storage.getElectricalPanels(parseInt(req.params.id));
      res.json(items);
    } catch (err: any) { res.status(500).json({ message: err.message }); }
  });
  app.post("/api/projects/:id/electrical/panels", async (req, res) => {
    try {
      const item = await storage.createElectricalPanel(parseInt(req.params.id), req.body);
      res.json(item);
    } catch (err: any) { res.status(500).json({ message: err.message }); }
  });
  app.patch("/api/electrical/panels/:id", async (req, res) => {
    try {
      const item = await storage.updateElectricalPanel(parseInt(req.params.id), req.body);
      res.json(item);
    } catch (err: any) { res.status(500).json({ message: err.message }); }
  });
  app.delete("/api/electrical/panels/:id", async (req, res) => {
    try {
      await storage.deleteElectricalPanel(parseInt(req.params.id));
      res.json({ success: true });
    } catch (err: any) { res.status(500).json({ message: err.message }); }
  });

  // ─── Electrical Circuits ───────────────────────────────────────────────────
  app.get("/api/projects/:id/electrical/circuits", async (req, res) => {
    try {
      const panelId = req.query.panelId ? parseInt(req.query.panelId as string) : undefined;
      const items = await storage.getElectricalCircuits(parseInt(req.params.id), panelId);
      res.json(items);
    } catch (err: any) { res.status(500).json({ message: err.message }); }
  });
  app.post("/api/projects/:id/electrical/circuits", async (req, res) => {
    try {
      const item = await storage.createElectricalCircuit(parseInt(req.params.id), req.body);
      res.json(item);
    } catch (err: any) { res.status(500).json({ message: err.message }); }
  });
  app.patch("/api/electrical/circuits/:id", async (req, res) => {
    try {
      const item = await storage.updateElectricalCircuit(parseInt(req.params.id), req.body);
      res.json(item);
    } catch (err: any) { res.status(500).json({ message: err.message }); }
  });
  app.delete("/api/electrical/circuits/:id", async (req, res) => {
    try {
      await storage.deleteElectricalCircuit(parseInt(req.params.id));
      res.json({ success: true });
    } catch (err: any) { res.status(500).json({ message: err.message }); }
  });

  // ─── Plumbing Fixtures ─────────────────────────────────────────────────────
  app.get("/api/projects/:id/plumbing", async (req, res) => {
    try {
      const items = await storage.getPlumbingFixtures(parseInt(req.params.id));
      res.json(items);
    } catch (err: any) { res.status(500).json({ message: err.message }); }
  });
  app.post("/api/projects/:id/plumbing", async (req, res) => {
    try {
      const item = await storage.createPlumbingFixture(parseInt(req.params.id), req.body);
      res.json(item);
    } catch (err: any) { res.status(500).json({ message: err.message }); }
  });
  app.patch("/api/plumbing/:id", async (req, res) => {
    try {
      const item = await storage.updatePlumbingFixture(parseInt(req.params.id), req.body);
      res.json(item);
    } catch (err: any) { res.status(500).json({ message: err.message }); }
  });
  app.delete("/api/plumbing/:id", async (req, res) => {
    try {
      await storage.deletePlumbingFixture(parseInt(req.params.id));
      res.json({ success: true });
    } catch (err: any) { res.status(500).json({ message: err.message }); }
  });

  // ─── Submittals ────────────────────────────────────────────────────────────
  app.get("/api/projects/:id/submittals", async (req, res) => {
    try {
      const items = await storage.getSubmittals(parseInt(req.params.id));
      res.json(items);
    } catch (err: any) { res.status(500).json({ message: err.message }); }
  });
  app.post("/api/projects/:id/submittals", async (req, res) => {
    try {
      const item = await storage.createSubmittal(parseInt(req.params.id), req.body);
      res.json(item);
    } catch (err: any) { res.status(500).json({ message: err.message }); }
  });
  app.patch("/api/submittals/:id", async (req, res) => {
    try {
      const item = await storage.updateSubmittal(parseInt(req.params.id), req.body);
      res.json(item);
    } catch (err: any) { res.status(500).json({ message: err.message }); }
  });
  app.delete("/api/submittals/:id", async (req, res) => {
    try {
      await storage.deleteSubmittal(parseInt(req.params.id));
      res.json({ success: true });
    } catch (err: any) { res.status(500).json({ message: err.message }); }
  });

  // Blueprint Analysis Report Download
  app.get("/api/reports/blueprint-analysis-dt10746", (req, res) => {
    const report = `================================================================================
BUILDMIND AI — CONSTRUCTION FIELD REPORT
PROJECT: Dollar Tree Store #10746 (Store #45117)
ADDRESS: 1029 E Grand Ave, Rothschild, WI 54476
PERMIT: #26-013 | DSPS: CB-012600013-PRBH
CURRENT REVISION: REV 4 (02/25/2026)
REPORT GENERATED: ${new Date().toLocaleDateString('en-US', {month:'long',day:'numeric',year:'numeric'})}
PREPARED BY: BuildMind AI | Superintendent Tool
================================================================================

SOURCE DOCUMENTS SCANNED
------------------------
- Original Construction Documents (39 pages, 10/30/2020 base set)
- REV #4 Sheets Only (10 pages, KLH Engineers, 2/25/2026)
- REV #4 Narrative (KLH Engineers Project #27191.00, 2/25/2026)
- All subcontractor NTPs (Laser Fire, Home Insulation, Seifert Electric,
  Best1 Plumbing, Central Temp)
- S2 Roof Framing Plan (field-verified from photo)
- EM-102 Energy Management Single Line Diagram
- EM-103 Interface Panel & eSCi Wiring Details

================================================================================
SECTION 1: RTU / HVAC UNIT INVENTORY (AUTHORITATIVE)
Source: S2 Roof Framing Plan (field photo confirmed)
NOTE: RTU = HVAC = same unit. "X" prefix (XRTU) = Existing unit, to remain.
================================================================================

| Unit   | Weight    | Status              | Location           | Structural Support         |
|--------|-----------|---------------------|--------------------|----------------------------|
| RTU-1  | 1,250 LBS | REPLACED — new unit | EXISTING LOCATION  | W24 STL. BEAM + DBL. JOIST |
| RTU-2  | 1,250 LBS | REPLACED — new unit | *** NEW LOCATION *** | W27 STL. BEAM + DBL. JOIST |
| XRTU-3 | —         | STAYS — service only| EXIST. TO REMAIN   | W27 STL. BEAM              |
| RTU-4  | 755 LBS   | REPLACED — new unit | EXISTING LOCATION  | W24 STL. BEAM + DBL. JOIST |
| XRTU-5 | —         | STAYS — service only| EXIST. TO REMAIN   | Bottom of building          |

3 Owner-Furnished Replacement Units:  RTU-1, RTU-2, RTU-4 (Central Temp installs)
2 Existing-to-Remain (service/commission only): XRTU-3, XRTU-5

RTU-1  = HVAC-1 on EMS drawings (EM-102)
RTU-2  = HVAC-2 on EMS drawings (new location — see RTU-2 relocation impacts below)
XRTU-3 = HVAC-3 on EMS drawings (existing, service only)
RTU-4  = HVAC-4 on EMS drawings
XRTU-5 = HVAC-5 on EMS drawings (existing, service only)

RTU-2 RELOCATION IMPACTS (4 trades affected):
  - Home Insulation: Patch old curb opening + cut new penetration (GC expense)
  - Best1 Plumbing:  Re-route gas branch line to new RTU-2 position
  - Seifert Electric: Re-route RTU-2 circuit + new disconnect at new roof position
  - Central Temp:    New duct connections to new roof opening; demo old duct stubs;
                     new condensate drain trap and routing at new curb location

================================================================================
SECTION 2: COMPLETE CHANGE LIST — ORIGINAL DOCS vs. REV #4
================================================================================

SHEET S2 — ROOF FRAMING PLAN
  [CHANGE] RTU-2 relocated to new position — W27 STL. BEAM required at new location
  [CHANGE] RTU-4 confirmed on drawings as replacement unit (755 LBS, W24 beam)
  [CHANGE] XRTU-3 designated EXIST. TO REMAIN (service scope only — not in NTP)
  [CHANGE] XRTU-5 designated EXIST. TO REMAIN (service scope only — not in NTP)
  [CLOUD Δ1] RTU-4 area — structural revision
  [CLOUD Δ2] XRTU-3 area — structural revision
  S2 GENERAL NOTES (verbatim):
    "COORDINATE THE EXACT LOCATION OF MECHANICAL ROOF TOP UNITS WITH
     MECHANICAL DRAWINGS AND ARCHITECTURAL DRAWINGS."
    "FIELD VERIFY ALL EXISTING CONDITIONS, DIMENSIONS AND ELEVATIONS.
     ALL DISCREPANCIES SHALL BE BROUGHT TO THE ATTENTION OF THE ARCHITECT."

SHEET M-101 — MECHANICAL HVAC FLOOR PLAN
  [CHANGE] RTU-1 location dimensions updated
  [CHANGE] RTU-2 new location — duct connections re-routed
  [CHANGE] RTU-3 (XRTU-3) coordinates corrected
  [CHANGE] XRTU-5 added to floor plan with service annotation
  [CHANGE] New supply duct routing to RTU-2 new position (approx. 12"x10" supply)
  [CHANGE] New return duct added near RTU-2 (approx. 14"x8")
  [CHANGE] Old duct stubs to original RTU-2 position — DEMO (dashed lines)
  [REVISION CLOUD] Near RTU-1/RTU-2 area — duct layout + connection point changes
  NOTE: Net ductwork change = MORE duct being added than removed.
        RTU-2 relocation drives new main trunk runs to new roof penetration points.

SHEET M-201 — MECHANICAL SCHEDULES
  [CHANGE] RTU schedule updated — XRTU-5 added as existing unit
  [CHANGE] RTU-4 added to replacement schedule
  [CHANGE] RTU-1 schedule spec updated (original: 6 Tons, 2000 CFM, 90 MBH gas,
           208V/3ph, 30A MCA, 40A OCP)
  NOTE: M-201 schedule page was not fully legible at scan resolution —
        pull hard copy for exact RTU-1/2/4 final specs

SHEET M-402 — MECHANICAL SEQUENCE OF OPERATIONS
  [CHANGE] XRTU-5 added to RTU fan mode, scheduling, and setpoint sequences
  [CHANGE] EMS (CYLON/EcoStruxure) must include XRTU-5 control points
  NOTE: M-402 references EMS integration for CO2 sensors and damper control

SHEET P-101 — PLUMBING WATER & GAS PLAN
  [NO CHANGE in REV #4]
  Original P-101 pipe sizes confirmed:
    Cold water (CW): 1-1/2", 1", 3/4" copper
    Hot water (HW):  1-1/2", 1", 3/4" copper
    Gas main:        1-1/4" steel
    Vent stacks:     2" and 3" PVC
    Sanitary:        PVC
  COORDINATION NOTE: RTU-2 gas branch must be re-routed — confirm 1-1/4" main
  has adequate capacity for extended branch run to new RTU-2 position.

SHEET P-201 — PLUMBING SCHEDULES
  [NO CHANGE in REV #4]
  Active schedules: Gas Load, Trap Primer, Floor Drain, Expansion Tank,
  Mop Sink, Lavatory, Drinking Fountain, Water Closet, HW Circulation Pump,
  Electric Water Heater.

SHEET E-102 — ELECTRICAL POWER PLAN
  [CHANGE] Delta Δ3 — Electric Equipment Supply Schedule revised
  [CHANGE] RTU-2 circuit re-routed to new physical location
  [CHANGE] XRTU-5 circuit added to panel schedule
  [CHANGE] Disconnect locations updated for RTU-2 new position
  NOTE: Refrigeration circuits unchanged in this revision

SHEET E-202 — ELECTRICAL SINGLE LINE / PANEL SCHEDULES
  [CHANGE — CLOUD 1] Panel A and/or B schedule — circuit arrangement modified
                      (RTU-1 MCA/OCP updated per REV #4)
  [CHANGE — CLOUD 2] Circuit load calculations updated — fault current ratings
                      re-checked after RTU-1 circuit change

SHEET EM-102 — ENERGY MANAGEMENT SINGLE LINE (ORIGINAL, NOT UPDATED)
  *** CRITICAL: ORIGINAL DOCS SHOW SimpleStat/eSCi SYSTEM ***
  *** REV #4 IDENTIFIES EMS AS CYLON BY SCHNEIDER ELECTRIC ***
  *** EM-102 AND EM-103 HAVE NOT BEEN UPDATED TO SHOW CYLON ***
  Unit labels on EM-102: HVAC-1 through HVAC-5 = RTU-1 through RTU-5
  CO2 sensors shown — still required under current design
  Remote Space Temperature Sensors (STS) on downrods — still required
  EMS panel located in utility/storage area

SHEET FPD1 — FIRE SPRINKLER DEMOLITION
  Full sprinkler system demo: all branch piping, cross mains, and sprinkler
  heads in renovation area to be removed.
  Main risers to remain.
  NOTE: Laser Fire's DSPS submittal required BEFORE installation (>20 heads)

SHEET FP2 — FIRE PROTECTION DETAILS & GENERAL NOTES
  "The contractor shall provide a complete and operational sprinkler system
   in accordance with NFPA 13 and all applicable codes."
  Upright sprinkler mounting details shown.
  NO hazard classification explicitly stated — must be declared in DSPS submittal.
  NOTE: Dry pendant heads required in cooler/freezer areas (Laser Fire scope).

SHEET D1 — DEMOLITION PLAN
  General Demolition Note #11: "General Construction to remove and replace
  fire-rated fire separation assemblies as indicated."
  GC responsible for pre-demo coordination across all trades.
  All removal to be documented in writing.

SHEET S1 — STRUCTURAL NOTES & FOUNDATION PLAN
  Slab infill: #4 dowels @ 12" OC each way, epoxy-set 6" deep into existing slab
  Concrete to match existing thickness
  Utility trench: reinforced concrete encasement around piping
  Minimum compacted fill: 6" each side of trench wall
  Concrete f'c = 4,000 PSI minimum

SHEET EN-101 — ENERGY LIGHTING COMPLIANCE
  References lighting compliance certification — responsibility unclear
  (A2 RCP notes lighting as "tenant-supplied electrical engineering technology")

================================================================================
SECTION 3: CRITICAL ISSUES — REQUIRES IMMEDIATE ENGINEERING ATTENTION
================================================================================

ISSUE #1 — EMS SYSTEM NOT UPDATED ON DRAWINGS [STOP-WORK RISK]
  Original EM-102/EM-103: SimpleStat thermostat + eSCi interface panel
  REV #4 / Field confirmation: CYLON by Schneider Electric (EcoStruxure)
  Status: EM-102 and EM-103 have NOT been updated to show CYLON wiring.
  Seifert Electric and Central Temp are working from outdated EMS drawings.
  ACTION: KLH or EMS engineer must issue revised EM-102/EM-103 showing
          CYLON controller wiring, points list, and interface details
          BEFORE Seifert begins any EMS wiring.
  RFI #53 OPEN

ISSUE #2 — ELECTRICAL CODE MISMATCH [HIGH RISK]
  E-001 design criteria references: 2017 NEC + ASHRAE 90.1-2013
  Project is permitted under: 2023 NEC + Wisconsin SPS + 2021 IECC
  Key differences requiring review:
    - NEC 210.8(F): GFCI at all outdoor HVAC equipment receptacles
      (RTU-1, RTU-2, RTU-4, XRTU-3, XRTU-5) — September 1, 2026 deadline
    - NEC 230.85: Emergency disconnects at meter (verify compliance)
    - Updated arc-fault and ground-fault requirements throughout
  ACTION: Seifert Electric must reverify all work against 2023 NEC.
          Add GFCI at all 5 RTU/HVAC service locations.

ISSUE #3 — XRTU-3 AND XRTU-5 SERVICE SCOPE NOT IN CENTRAL TEMP NTP
  Central Temp NTP ($36,778): 3 replacement RTUs only
  S2 confirms: XRTU-3 AND XRTU-5 are existing-to-remain, service only
  M-402 REV #4: XRTU-5 added to EMS sequence of operations
  Both XRTU-3 and XRTU-5 need: inspection, full service, M09 checklist,
  EMS commissioning, and startup verification
  ACTION: Call Chad DuFrane (Central Temp, 920-731-5071) — confirm whether
          XRTU-3 and XRTU-5 service is in $36,778 or requires 2 change orders.
  RFI #51 OPEN (now covers BOTH units)

ISSUE #4 — RTU-4 NEW UNIT WEIGHT vs. EXISTING 755 LBS STRUCTURAL CAPACITY
  S2 shows existing RTU-4 at 755 LBS, W24 STL. BEAM
  RTU-4 is being replaced with owner-furnished unit
  If new unit weighs more than 755 LBS, W24 beam may be undersized
  ACTION: Get RTU-4 equipment submittal from Dollar Tree/Sun PM immediately.
          Structural engineer to confirm W24 beam capacity before curb installation.
  RFI #55 OPEN

ISSUE #5 — OWNER-FURNISHED RTUs MUST HAVE ECONOMIZERS
  Per IECC C403.5 / SPS 363.0403(3): All new RTUs require:
    - Economizer
    - Fault detection
    - Class I motorized damper
  Original M-201 RTU-1 spec: 6 Tons, 2000 CFM, 90 MBH, 208V/3ph, 30A MCA, 40A OCP
    — NO economizer explicitly called out on original schedule
  RTU-1, RTU-2, RTU-4 are all owner-furnished — Dollar Tree supplies them
  ACTION: Get equipment submittals from Dollar Tree for RTU-1, RTU-2, RTU-4.
          Verify economizer + Class I damper + fault detection included.
          If not, DSPS will reject at inspection.
  RFI #39 OPEN / RFI #56 OPEN

ISSUE #6 — CONDENSATE DRAIN RELOCATION AT RTU-2 NEW POSITION
  RTU-2 moving = condensate drain must be re-routed
  New trap required at new curb location
  No sheet explicitly assigns this work
  Potential for roof ponding if old drain not properly abandoned
  ACTION: Central Temp + Best1 + Home Insulation — coordinate condensate
          drain routing, trap installation, and roof membrane penetration
          for new RTU-2 position before curb installation.
  RFI #57 OPEN

ISSUE #7 — FIRE SPRINKLER HAZARD CLASSIFICATION NOT STATED
  FP2 requires NFPA 13 compliance but no hazard class declared
  Dollar Tree: Light Hazard (sales), Ordinary Hazard Group 1 (stockroom)
  Incorrect classification = hydraulic calc rejection by DSPS
  ACTION: Laser Fire to explicitly state hazard classification in DSPS submittal.
          Confirm with DSPS reviewer Jason Hansen (920-492-7728) before submitting.
          David Bartolerio (Laser Fire, 608-205-7219, WI Lic #948334).
  RFI #58 OPEN

================================================================================
SECTION 4: OPEN RFI SUMMARY (41 Total — 39 Open, 2 Reviewed)
================================================================================

KEY OPEN RFIs:
  #39  Economizer required on ALL new RTUs (IECC C403.5 / SPS 363)
  #51  XRTU-3 AND XRTU-5 service scope — confirm both covered or 2 COs needed
  #52  Seifert to re-verify RTU-1/RTU-2 circuits vs. E-102/E-202 REV #4
  #53  NEW: EM-102/EM-103 not updated for CYLON — drawings required before wiring
  #54  CLOSED: HVAC-1 thru HVAC-5 = RTU-1 thru RTU-5 (numbering confirmed 1:1)
  #55  NEW: RTU-4 new unit weight vs. W24 beam capacity
  #56  NEW: Economizer/damper confirmation on owner-furnished RTU-1, RTU-2, RTU-4
  #57  NEW: Condensate drain relocation at RTU-2 new position
  #58  NEW: Laser Fire hazard classification for DSPS hydraulic calcs

================================================================================
SECTION 5: SUBCONTRACTOR CONTACT DIRECTORY
================================================================================

GENERAL / MANAGEMENT
  Sun PM (GC):          T.C. DiYanni
  Sun CM:               Will Sparks — 281-706-5894
  Dollar Tree PM:       Miguel Sanchez Leos
  CCI:                  Kyle Spangler — 314-991-2633
  M Architects (EOR):   Brian Eady — brian@marchitects.com | 586-933-3010

SUBCONTRACTORS
  Laser Fire (Fire Sprinkler): David Bartolerio
                                608-205-7219 | WI Lic #948334
                                NTP: $17,100 | Signed: 1/26/2026
  Home Insulation (Roofing):   Zachary Chaignot
                                715-359-6505
                                NTP: $2,935 | Signed: 1/27/2026
  Seifert Electric:            Kim Kluz — 715-693-2625
                                NTP: $45,450 | Signed: 1/30/2026
  Best1 Plumbing:              Zach/Nate Roth — 715-241-0883
                                NTP: $25,415 | Signed: 2/3/2026
  Central Temp (HVAC):         Chad DuFrane — 920-731-5071
                                NTP: $36,778 | Signed: 2/19/2026
                                PAYMENT: Net 10 days from invoice (URGENT)

INSPECTORS / AUTHORITIES
  DSPS Plan Reviewer:   Jason Hansen — 920-492-7728 | jason.hansen@wisconsin.gov
  DIS Inspector:        Jon Molledahl — 608-225-6520 | jon.molledahl@wisconsin.gov
  Municipal Clerk:      Elizabeth Felkner (Rothschild) — 715-359-3660

================================================================================
SECTION 6: PROJECT SUMMARY
================================================================================

  Permit:          #26-013 | Fee paid: $2,517.76
  DSPS Approval:   CB-012600013-PRBH (1/13/2026, expires 1/13/2028)
  Construction:    03/09/2026 – 04/17/2026 (~6 weeks)
  Building:        11,252 SF | 1 Story | Type II-B | Fully Sprinklered
  Occupancy:       M - Mercantile | 156 persons (152 sales + 4 stockroom)
  Service:         2 x 200A overhead | Est. project cost: $377,000
  Total Subcontracts: ~$127,678
  EMS System:      CYLON by Schneider Electric (EcoStruxure)
  Asbestos:        1,500 SF black mastic ACM (Vertex #102403) — landlord remediating
  GFCI Deadline:   September 1, 2026 (NEC 210.8(F) outdoor HVAC receptacles)

================================================================================
END OF REPORT
BuildMind AI | buildmind.replit.app
Report covers: Blueprint Scan + REV #4 Cross-Reference + RFI Status
================================================================================
`;

    res.setHeader('Content-Type', 'text/plain; charset=utf-8');
    res.setHeader('Content-Disposition', 'attachment; filename="DT10746-Rothschild-Blueprint-Report.txt"');
    res.send(report);
  });

  // Blueprint Analysis Report — Word (.docx) Download
  app.get("/api/reports/blueprint-analysis-dt10746/download", async (req, res) => {
    try {
      const { Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell, WidthType, BorderStyle, AlignmentType, ShadingType } = await import("docx") as any;

      const red = "C0392B";
      const orange = "D35400";
      const navy = "1A3A5C";
      const gray = "F2F4F7";

      const h1 = (text: string) => new Paragraph({ text, heading: HeadingLevel.HEADING_1, spacing: { before: 400, after: 120 } });
      const h2 = (text: string, color = navy) => new Paragraph({
        children: [new TextRun({ text, bold: true, size: 26, color })],
        spacing: { before: 320, after: 100 },
        border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "CCCCCC" } }
      });
      const h3 = (text: string) => new Paragraph({
        children: [new TextRun({ text, bold: true, size: 22, color: navy })],
        spacing: { before: 240, after: 80 }
      });
      const p = (text: string) => new Paragraph({ text, spacing: { after: 80 } });
      const bullet = (text: string) => new Paragraph({ text, bullet: { level: 0 }, spacing: { after: 60 } });
      const subbullet = (text: string) => new Paragraph({ text, bullet: { level: 1 }, spacing: { after: 40 } });
      const bold = (label: string, value: string) => new Paragraph({
        children: [new TextRun({ text: label, bold: true }), new TextRun({ text: value })],
        spacing: { after: 60 }
      });
      const flagPara = (text: string, color: string) => new Paragraph({
        children: [new TextRun({ text, color, bold: true })],
        spacing: { after: 80 }
      });
      const divider = () => new Paragraph({ text: "", border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: "DDDDDD" } }, spacing: { before: 200, after: 200 } });

      const makeTable = (headers: string[], rows: string[][], colWidths?: number[]) => {
        const widths = colWidths || headers.map(() => Math.floor(9000 / headers.length));
        const headerRow = new TableRow({
          children: headers.map((h, i) => new TableCell({
            children: [new Paragraph({ children: [new TextRun({ text: h, bold: true, color: "FFFFFF" })] })],
            width: { size: widths[i], type: WidthType.DXA },
            shading: { type: ShadingType.SOLID, fill: navy }
          }))
        });
        const dataRows = rows.map((row, ri) => new TableRow({
          children: row.map((cell, ci) => new TableCell({
            children: [new Paragraph({ children: [new TextRun({ text: cell, size: 18 })] })],
            width: { size: widths[ci], type: WidthType.DXA },
            shading: { type: ShadingType.SOLID, fill: ri % 2 === 0 ? "FFFFFF" : gray }
          }))
        }));
        return new Table({ rows: [headerRow, ...dataRows], width: { size: 9000, type: WidthType.DXA } });
      };

      const today = new Date().toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' });

      const doc = new Document({
        styles: {
          default: {
            document: { run: { font: "Calibri", size: 22 } }
          }
        },
        sections: [{
          properties: { page: { margin: { top: 1080, bottom: 1080, left: 1080, right: 1080 } } },
          children: [

            // Cover Header
            new Paragraph({
              children: [new TextRun({ text: "BUILDMIND AI — CONSTRUCTION FIELD REPORT", bold: true, size: 36, color: navy })],
              alignment: AlignmentType.CENTER, spacing: { after: 160 }
            }),
            new Paragraph({
              children: [new TextRun({ text: "Dollar Tree Store #10746 (Store #45117)", bold: true, size: 28, color: "444444" })],
              alignment: AlignmentType.CENTER, spacing: { after: 80 }
            }),
            new Paragraph({
              children: [new TextRun({ text: "1029 E Grand Ave, Rothschild, WI 54476", size: 24, color: "666666" })],
              alignment: AlignmentType.CENTER, spacing: { after: 80 }
            }),
            new Paragraph({
              children: [new TextRun({ text: `Report Generated: ${today}  |  Permit #26-013  |  REV 4 — 02/25/2026`, size: 20, color: "888888" })],
              alignment: AlignmentType.CENTER, spacing: { after: 320 }
            }),
            divider(),

            // Source Documents
            h2("SOURCE DOCUMENTS SCANNED"),
            bullet("Original Construction Documents — 39 pages (10/30/2020 base set)"),
            bullet("REV #4 Sheets Only — 10 pages (KLH Engineers, 2/25/2026)"),
            bullet("REV #4 Narrative — KLH Engineers Project #27191.00, 2/25/2026"),
            bullet("All Subcontractor NTPs — Laser Fire, Home Insulation, Seifert Electric, Best1 Plumbing, Central Temp"),
            bullet("S2 Roof Framing Plan — field-verified from photo"),
            bullet("EM-102 Energy Management Single Line Diagram"),
            bullet("EM-103 Interface Panel & eSCi Wiring Details"),
            divider(),

            // Section 1 — RTU Inventory
            h1("SECTION 1 — RTU / HVAC UNIT INVENTORY"),
            new Paragraph({
              children: [
                new TextRun({ text: "TERMINOLOGY: ", bold: true }),
                new TextRun({ text: 'RTU = HVAC = same unit.  "X" prefix (XRTU) = Existing unit, to remain.  HVAC-1 through HVAC-5 on EMS drawings = RTU-1 through RTU-5 (1:1 confirmed).' })
              ], spacing: { after: 200 }
            }),
            makeTable(
              ["Unit", "Weight", "Status", "Location", "Structural Support"],
              [
                ["RTU-1",  "1,250 LBS", "REPLACED — new unit",    "Existing location",      "W24 STL. BEAM + DBL. JOIST"],
                ["RTU-2",  "1,250 LBS", "REPLACED — new unit",    "★ NEW LOCATION",         "W27 STL. BEAM + DBL. JOIST"],
                ["XRTU-3", "—",         "STAYS — service only",   "EXIST. TO REMAIN",       "W27 STL. BEAM"],
                ["RTU-4",  "755 LBS",   "REPLACED — new unit",    "Existing location",      "W24 STL. BEAM + DBL. JOIST"],
                ["XRTU-5", "—",         "STAYS — service only",   "EXIST. TO REMAIN (bottom)", "—"],
              ],
              [1200, 1200, 2000, 2200, 2400]
            ),
            p(""),
            bold("3 Owner-Furnished Replacements (Central Temp installs): ", "RTU-1, RTU-2, RTU-4"),
            bold("2 Existing-to-Remain (service/commission only): ", "XRTU-3, XRTU-5"),

            h3("RTU-2 Relocation — 4 Trades Affected"),
            bullet("Home Insulation: Patch old curb opening + cut new roof penetration (GC expense)"),
            bullet("Best1 Plumbing: Re-route gas branch line to new RTU-2 position"),
            bullet("Seifert Electric: Re-route RTU-2 circuit + new disconnect at new roof position"),
            bullet("Central Temp: New duct connections to new opening; demo old duct stubs; new condensate drain trap at new curb location"),
            divider(),

            // Section 2 — Change List
            h1("SECTION 2 — COMPLETE CHANGE LIST: ORIGINAL DOCS vs. REV #4"),

            h3("S2 — Roof Framing Plan"),
            bullet("RTU-2 relocated to new position — W27 STL. BEAM required at new location  [Δ cloud]"),
            bullet("RTU-4 confirmed on drawings as replacement unit (755 LBS, W24 beam)  [Δ1 cloud]"),
            bullet("XRTU-3 designated EXIST. TO REMAIN — service scope only, not in NTP  [Δ2 cloud]"),
            bullet("XRTU-5 designated EXIST. TO REMAIN — service scope only, not in NTP"),
            subbullet("S2 Note: 'COORDINATE EXACT LOCATION OF RTUs WITH MECHANICAL AND ARCHITECTURAL DRAWINGS'"),
            subbullet("S2 Note: 'FIELD VERIFY ALL EXISTING CONDITIONS AND ELEVATIONS'"),

            h3("M-101 — Mechanical HVAC Floor Plan"),
            bullet("RTU-1 location dimensions updated"),
            bullet("RTU-2 new location — duct connections re-routed to new roof opening"),
            bullet("RTU-3 (XRTU-3) coordinates corrected on plan"),
            bullet("XRTU-5 added to floor plan with service annotation"),
            bullet("NEW supply duct routing to RTU-2 new position (~12\"x10\" supply)"),
            bullet("NEW return duct added near RTU-2 (~14\"x8\")"),
            bullet("Old duct stubs to original RTU-2 position — DEMO (dashed lines)"),
            bullet("NET DUCTWORK CHANGE: More duct being added than removed — RTU-2 relocation drives new main trunk runs"),

            h3("M-201 — Mechanical Schedules"),
            bullet("RTU schedule updated — XRTU-5 added as existing unit"),
            bullet("RTU-4 added to replacement schedule"),
            bullet("RTU-1 original spec: 6 Tons, 2,000 CFM, 90 MBH gas, 208V/3ph, 30A MCA, 40A OCP"),
            bullet("NOTE: Pull hard copy M-201 REV #4 for final confirmed specs on all RTUs"),

            h3("M-402 — Sequence of Operations"),
            bullet("XRTU-5 added to RTU fan mode, scheduling, and setpoint sequences"),
            bullet("EMS (CYLON/EcoStruxure) programming must include XRTU-5 control points"),

            h3("E-102 — Electrical Power Plan"),
            bullet("Delta Δ3 — Electric Equipment Supply Schedule revised"),
            bullet("RTU-2 circuit re-routed to new physical location"),
            bullet("XRTU-5 circuit added to panel schedule"),
            bullet("Disconnect locations updated for RTU-2 new position"),

            h3("E-202 — Electrical Single Line / Panel Schedules"),
            bullet("CLOUD 1: Panel A/B schedule circuit arrangement modified (RTU-1 MCA/OCP updated)"),
            bullet("CLOUD 2: Circuit load calculations updated — fault current re-checked after RTU-1 change"),

            h3("EM-102 / EM-103 — EMS Drawings (CRITICAL — NOT UPDATED)"),
            flagPara("★ ORIGINAL DOCS SHOW SimpleStat/eSCi SYSTEM — REV #4 IS CYLON BY SCHNEIDER ELECTRIC", red),
            flagPara("★ EM-102 AND EM-103 HAVE NOT BEEN REVISED — SEIFERT IS WORKING FROM WRONG EMS DRAWINGS", red),
            bullet("HVAC-1 through HVAC-5 on EM-102 = RTU-1 through RTU-5 (numbering confirmed 1:1)"),
            bullet("CO2 sensors required — still shown on EM drawings"),
            bullet("Remote Space Temperature Sensors (STS) on downrods — still required"),

            h3("P-101 — Plumbing (No Changes in REV #4)"),
            bullet("Cold/Hot water: 1-1/2\", 1\", 3/4\" copper"),
            bullet("Gas main: 1-1/4\" steel — RTU-2 branch must be re-routed to new position"),
            bullet("Vent stacks: 2\" and 3\" PVC"),

            h3("FPD1 — Fire Sprinkler Demo / FP2 — Fire Protection"),
            bullet("Full sprinkler system demo: all branch piping, cross mains, heads removed in renovation area"),
            bullet("Main risers remain"),
            bullet("FP2: NFPA 13 compliance required — hazard classification NOT explicitly stated on drawings"),
            bullet("Dry pendant heads required in cooler/freezer (Laser Fire scope)"),
            bullet("DSPS submittal required before installation (>20 heads, WI SPS 314)"),
            divider(),

            // Section 3 — Critical Issues
            h1("SECTION 3 — CRITICAL ISSUES — IMMEDIATE ENGINEERING ATTENTION REQUIRED"),

            new Paragraph({
              children: [new TextRun({ text: "🔴  CRITICAL — Stop-Work Level", bold: true, size: 26, color: red })],
              spacing: { before: 200, after: 120 }
            }),

            h3("ISSUE #1 — EMS System Drawings Not Updated for CYLON"),
            flagPara("EM-102 and EM-103 still show SimpleStat/eSCi. Project EMS is CYLON by Schneider Electric (EcoStruxure). Seifert Electric is wiring from outdated drawings.", red),
            bold("Action: ", "KLH or EMS engineer must issue revised EM-102/EM-103 showing CYLON controller wiring, points list, and interface details BEFORE Seifert begins any EMS wiring."),
            bold("RFI: ", "#53 OPEN"),

            h3("ISSUE #2 — Electrical Design Criteria References Wrong Code"),
            flagPara("E-001 references 2017 NEC + ASHRAE 90.1-2013. Project is permitted under 2023 NEC + Wisconsin SPS + 2021 IECC.", orange),
            bullet("NEC 210.8(F): GFCI required at all outdoor HVAC equipment — September 1, 2026 deadline"),
            bullet("NEC 230.85: Emergency disconnect at meter — verify compliance"),
            bullet("Updated arc-fault and ground-fault requirements throughout"),
            bold("Action: ", "Seifert Electric must reverify all work against 2023 NEC. Add GFCI at all 5 RTU/HVAC service locations."),

            new Paragraph({
              children: [new TextRun({ text: "🟠  HIGH PRIORITY — Resolve Before Trade Starts", bold: true, size: 26, color: orange })],
              spacing: { before: 240, after: 120 }
            }),

            h3("ISSUE #3 — XRTU-3 AND XRTU-5 Service Scope Not in Central Temp NTP"),
            flagPara("Central Temp NTP covers 3 replacement RTUs only. BOTH XRTU-3 and XRTU-5 are existing-to-remain, service only — neither is in the $36,778 contract.", orange),
            bold("Action: ", "Call Chad DuFrane (Central Temp) 920-731-5071 — confirm whether service of XRTU-3 AND XRTU-5 is included or requires 2 change orders."),
            bold("RFI: ", "#51 OPEN — now covers BOTH units"),

            h3("ISSUE #4 — RTU-4 New Unit Weight vs. Existing 755 LBS Structural Capacity"),
            flagPara("S2 shows existing RTU-4 at 755 LBS on W24 beam. If new owner-furnished RTU-4 weighs more, W24 beam is undersized.", orange),
            bold("Action: ", "Get RTU-4 equipment submittal from Dollar Tree/Sun PM immediately. Structural engineer to confirm W24 beam before curb installation."),
            bold("RFI: ", "#55 OPEN"),

            h3("ISSUE #5 — Economizer Not Confirmed on Owner-Furnished RTUs"),
            flagPara("IECC C403.5 / SPS 363.0403(3): ALL new RTUs require economizer + fault detection + Class I motorized damper. Dollar Tree supplies RTU-1, RTU-2, RTU-4 — no economizer confirmed in specs.", orange),
            bold("Action: ", "Get equipment submittals for RTU-1, RTU-2, RTU-4 from Dollar Tree NOW. Verify before units ship."),
            bold("RFI: ", "#39 / #56 OPEN"),

            h3("ISSUE #6 — Condensate Drain Relocation at RTU-2 Not Assigned"),
            flagPara("RTU-2 moving = new condensate drain trap required at new curb position. No sheet assigns this work. Misrouted condensate = roof ponding and water intrusion.", orange),
            bold("Action: ", "Central Temp + Best1 + Home Insulation coordinate drain routing, trap installation, and roof membrane penetration before curb set."),
            bold("RFI: ", "#57 OPEN"),

            h3("ISSUE #7 — Fire Sprinkler Hazard Classification Not Stated"),
            flagPara("FP2 requires NFPA 13 compliance but no hazard class declared. Wrong classification = DSPS hydraulic calc rejection.", orange),
            bullet("Sales floor: Light Hazard — Stockroom: Ordinary Hazard Group 1"),
            bold("Action: ", "Laser Fire to explicitly state hazard classification in DSPS submittal. Confirm with Jason Hansen (DSPS) 920-492-7728 before submitting."),
            bold("RFI: ", "#58 OPEN"),
            divider(),

            // Section 4 — RFI Summary
            h1("SECTION 4 — OPEN RFI SUMMARY"),
            bold("Total: ", "41 RFIs  |  39 Open  |  2 Reviewed"),
            p(""),
            makeTable(
              ["RFI #", "Title", "Status", "Trade"],
              [
                ["#39",  "Economizer required on ALL new RTUs (IECC C403.5 / SPS 363)",              "OPEN",   "HVAC / DT"],
                ["#51",  "XRTU-3 AND XRTU-5 service scope — confirm both in NTP or 2 COs",          "OPEN",   "Central Temp"],
                ["#52",  "Seifert re-verify RTU-1/RTU-2 circuits vs. E-102/E-202 REV #4",            "OPEN",   "Seifert"],
                ["#53",  "NEW: EM-102/EM-103 not updated for CYLON — drawings required",              "OPEN",   "KLH / Seifert"],
                ["#54",  "HVAC-1 thru HVAC-5 = RTU-1 thru RTU-5 — numbering confirmed 1:1",          "CLOSED", "—"],
                ["#55",  "NEW: RTU-4 new unit weight vs. W24 beam capacity",                          "OPEN",   "Structural"],
                ["#56",  "NEW: Economizer/damper confirmation on owner-furnished RTU-1, RTU-2, RTU-4","OPEN",   "DT / KLH"],
                ["#57",  "NEW: Condensate drain relocation at RTU-2 new position",                    "OPEN",   "Central Temp / Best1"],
                ["#58",  "NEW: Laser Fire hazard classification for DSPS hydraulic calcs",            "OPEN",   "Laser Fire"],
              ],
              [800, 4200, 900, 2100]
            ),
            divider(),

            // Section 5 — Contacts
            h1("SECTION 5 — SUBCONTRACTOR CONTACTS"),
            h3("General / Management"),
            makeTable(
              ["Role", "Name", "Phone", "Email"],
              [
                ["GC Superintendent", "T.C. DiYanni (Sun PM)", "—", "—"],
                ["CM", "Will Sparks (Sun PM)", "281-706-5894", "—"],
                ["DT Project Mgr", "Miguel Sanchez Leos", "—", "—"],
                ["CCI", "Kyle Spangler", "314-991-2633", "—"],
                ["Architect (EOR)", "Brian Eady (M Architects)", "586-933-3010", "brian@marchitects.com"],
              ],
              [2000, 2200, 1800, 3000]
            ),
            p(""),
            h3("Subcontractors"),
            makeTable(
              ["Trade / Company", "Contact", "Phone", "NTP Amount", "Payment"],
              [
                ["Laser Fire (Sprinkler)", "David Bartolerio  |  WI Lic #948334", "608-205-7219", "$17,100", "Net 30"],
                ["Home Insulation (Roofing)", "Zachary Chaignot", "715-359-6505", "$2,935", "Net 30"],
                ["Seifert Electric", "Kim Kluz", "715-693-2625", "$45,450", "Net 30"],
                ["Best1 Plumbing", "Zach/Nate Roth", "715-241-0883", "$25,415", "Net 30"],
                ["Central Temp (HVAC)", "Chad DuFrane", "920-731-5071", "$36,778", "⚠ Net 10 days"],
              ],
              [2200, 2200, 1500, 1400, 1700]
            ),
            p(""),
            h3("Inspectors / Authorities"),
            makeTable(
              ["Role", "Name", "Phone", "Email"],
              [
                ["DSPS Plan Reviewer", "Jason Hansen", "920-492-7728", "jason.hansen@wisconsin.gov"],
                ["DIS Inspector", "Jon Molledahl", "608-225-6520", "jon.molledahl@wisconsin.gov"],
                ["Municipal Clerk (Rothschild)", "Elizabeth Felkner", "715-359-3660", "efelkner@rothschildwi.com"],
              ],
              [2200, 1800, 1800, 3200]
            ),
            divider(),

            // Section 6 — Project Summary
            h1("SECTION 6 — PROJECT SUMMARY"),
            makeTable(
              ["Field", "Value"],
              [
                ["Store",              "Dollar Tree #10746 (Store #45117)"],
                ["Address",            "1029 E Grand Ave, Rothschild, WI 54476"],
                ["Permit",             "#26-013  |  Fee paid: $2,517.76"],
                ["DSPS Approval",      "CB-012600013-PRBH (1/13/2026 — expires 1/13/2028)"],
                ["Construction Dates", "03/09/2026 – 04/17/2026 (~6 weeks)"],
                ["Building",           "11,252 SF  |  1 Story  |  Type II-B  |  Fully Sprinklered"],
                ["Occupancy",          "M - Mercantile  |  156 persons (152 sales + 4 stockroom)"],
                ["Electrical Service", "2 × 200A overhead"],
                ["Est. Project Cost",  "$377,000"],
                ["Total Subcontracts", "~$127,678"],
                ["EMS System",         "CYLON by Schneider Electric (EcoStruxure)"],
                ["Asbestos",           "1,500 SF black mastic ACM (Vertex #102403) — landlord remediating"],
                ["GFCI Deadline",      "September 1, 2026 — NEC 210.8(F) outdoor HVAC receptacles"],
                ["Electrical Code",    "2023 NEC  |  Energy: 2021 IECC / Wisconsin SPS"],
                ["Fire Code",          "NFPA 13 (2025)  |  NFPA 72 (2025)"],
              ],
              [3000, 6000]
            ),

            p(""),
            p(""),
            new Paragraph({
              children: [new TextRun({ text: "BuildMind AI  |  Construction Superintendent Tool  |  DT #10746 Rothschild WI", size: 18, color: "999999" })],
              alignment: AlignmentType.CENTER, spacing: { before: 400 }
            }),

          ]
        }]
      });

      const buffer = await Packer.toBuffer(doc);
      res.setHeader('Content-Type', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document');
      res.setHeader('Content-Disposition', 'attachment; filename="DT10746-Rothschild-Blueprint-Report.docx"');
      res.send(buffer);
    } catch (err: any) {
      res.status(500).json({ message: `Error generating Word document: ${err.message}` });
    }
  });

  // ─── Punchlist ────────────────────────────────────────────────────────────────
  app.get("/api/projects/:projectId/punchlist", async (req, res) => {
    try {
      const projectId = parseInt(req.params.projectId);
      const items = await storage.getPunchlistItems(projectId);
      res.json(items);
    } catch (err: any) {
      res.status(500).json({ message: err.message });
    }
  });

  app.post("/api/projects/:projectId/punchlist", async (req, res) => {
    try {
      const projectId = parseInt(req.params.projectId);
      const item = await storage.createPunchlistItem({ ...req.body, projectId, seeded: false });
      res.json(item);
    } catch (err: any) {
      res.status(500).json({ message: err.message });
    }
  });

  app.patch("/api/projects/:projectId/punchlist/:id", async (req, res) => {
    try {
      const id = parseInt(req.params.id);
      const item = await storage.updatePunchlistItem(id, req.body);
      res.json(item);
    } catch (err: any) {
      res.status(500).json({ message: err.message });
    }
  });

  app.delete("/api/projects/:projectId/punchlist/:id", async (req, res) => {
    try {
      const id = parseInt(req.params.id);
      await storage.deletePunchlistItem(id);
      res.json({ success: true });
    } catch (err: any) {
      res.status(500).json({ message: err.message });
    }
  });

  // Run seed after server is ready — non-blocking so startup is instant
  setTimeout(() => {
    seedDatabase().catch(err => console.error("[seed] Error:", err.message));
  }, 5000);

  // Backfill: auto-analyze any existing uploaded files that never got analyzed
  setTimeout(async () => {
    try {
      const projects = await storage.getProjects();
      for (const project of projects) {
        const docs = await storage.getDocuments(project.id);
        const unanalyzed = docs.filter(d =>
          !d.content ||
          d.content.startsWith("File uploaded") ||
          d.content.startsWith("Blueprint uploaded") ||
          d.content === "Analyzing document…"
        );
        if (unanalyzed.length > 0) {
          console.log(`[auto-analyze] Backfilling ${unanalyzed.length} document(s) for project #${project.id}`);
          for (const doc of unanalyzed) {
            await autoAnalyzeDocument(doc.id, project.id);
          }
        }
      }
    } catch (err: any) {
      console.error("[auto-analyze] Backfill error:", err.message);
    }
  }, 15000); // 15s delay — wait for server fully ready and seed to complete

  return httpServer;
}
