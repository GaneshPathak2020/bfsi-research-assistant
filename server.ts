import express from 'express';
import dotenv from 'dotenv';
import path from 'path';
import { fileURLToPath } from 'url';
import { createServer as createViteServer } from 'vite';
import { GoogleGenAI, Type } from '@google/genai';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT ? parseInt(process.env.PORT, 10) : 3000;
const isProduction = process.env.NODE_ENV === 'production';

app.use(express.json({ limit: '30mb' }));
app.use(express.urlencoded({ extended: true, limit: '30mb' }));

// Server-side Gemini initialization per instructions
const ai = new GoogleGenAI({
  apiKey: process.env.GEMINI_API_KEY,
  httpOptions: {
    headers: {
      'User-Agent': 'aistudio-build',
    },
  },
});

// Helper for safe JSON parse
function cleanAndParseJSON(rawText: string) {
  let cleaned = rawText.trim();
  if (cleaned.startsWith('```json')) {
    cleaned = cleaned.replace(/^```json\s*/, '').replace(/\s*```$/, '');
  } else if (cleaned.startsWith('```')) {
    cleaned = cleaned.replace(/^```\s*/, '').replace(/\s*```$/, '');
  }
  return JSON.parse(cleaned);
}

// 1. EXTRACT PAPER ROUTE
app.post('/api/extract-paper', async (req, res) => {
  try {
    const { text, fileData, mimeType, fileName } = req.body;

    if (!text && !fileData) {
      return res.status(400).json({ error: 'Please provide either paper text or an uploaded document.' });
    }

    const systemPrompt = `You are a rigorous academic research assistant specializing in Banking, Finance, and Insurance (BFSI) literature reviews.
Your goal is to extract empirical and conceptual details from the supplied academic paper with extreme fidelity.

CRITICAL RULES:
1. Include source-text references (verbatim quote and section name) wherever available in the text.
2. Mark missing information as "Not reported in the supplied paper."
3. DO NOT invent, hallucinate, extrapolate citations, sample sizes, findings, or research gaps. If a field is not present or unclear, explicitly write "Not reported in the supplied paper."
4. CLEARLY DISTINGUISH extracted facts (verbatim or directly attested statements with exact quotes) from AI-generated interpretations (theoretical implications, policy context, methodology critique).
5. Categorize the paper into one of: Banking, Finance, Insurance, FinTech, or Cross-BFSI.

Extract all fields according to the required JSON schema.`;

    const extractionInstruction = `Carefully analyze the supplied BFSI paper document/text and extract the literature review matrix parameters:
- Title, Authors, Year, Journal/Source, BFSI Domain
- Research Objective (with source quote and section)
- Methodology (econometric, statistical, empirical design, or theoretical modeling with source quote)
- Sample Details (dataset name, time period, geography/country, sample size N, with source quote)
- Variables:
  * Dependent variables (with proxies/measures and source quote)
  * Independent / Explanatory variables (with proxies/measures and source quote)
  * Control / Moderator variables (with proxies/measures and source quote)
- Key Findings (list 2-5 specific empirical results, statistical significance, effect sizes with exact quotes)
- Limitations (author-reported limitations or explicit study caveats with exact quotes. If authors did not report limitations in this excerpt, state "Not reported in the supplied paper.")
- Extracted Facts: 4 to 8 factual statements directly grounded in the text with exact quote and section.
- AI Interpretations: 2 to 4 analytical interpretations (e.g. theoretical framing, policy relevance, potential econometrics caveats) clearly marked as AI-generated inferences, grounded in the extracted facts.

Remember: If anything is missing in the supplied paper, mark it strictly as "Not reported in the supplied paper."`;

    let response;

    if (fileData && mimeType) {
      // Support PDF or other document inlineData
      response = await ai.models.generateContent({
        model: 'gemini-3.8-flash',
        contents: [
          {
            role: 'user',
            parts: [
              {
                inlineData: {
                  mimeType: mimeType,
                  data: fileData,
                },
              },
              {
                text: `${extractionInstruction}\nFilename: ${fileName || 'Uploaded Document'}`,
              },
            ],
          },
        ],
        config: {
          systemInstruction: systemPrompt,
          responseMimeType: 'application/json',
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              title: { type: Type.STRING },
              authors: {
                type: Type.ARRAY,
                items: { type: Type.STRING },
              },
              year: { type: Type.STRING },
              journalOrSource: { type: Type.STRING },
              bfsiDomain: {
                type: Type.STRING,
                description: 'Must be one of: Banking, Finance, Insurance, FinTech, Cross-BFSI',
              },
              researchObjective: {
                type: Type.OBJECT,
                properties: {
                  value: { type: Type.STRING },
                  isReported: { type: Type.BOOLEAN },
                  sourceQuote: { type: Type.STRING },
                  section: { type: Type.STRING },
                },
                required: ['value', 'isReported'],
              },
              methodology: {
                type: Type.OBJECT,
                properties: {
                  value: { type: Type.STRING },
                  isReported: { type: Type.BOOLEAN },
                  sourceQuote: { type: Type.STRING },
                  section: { type: Type.STRING },
                },
                required: ['value', 'isReported'],
              },
              sample: {
                type: Type.OBJECT,
                properties: {
                  isReported: { type: Type.BOOLEAN },
                  description: { type: Type.STRING },
                  dataset: { type: Type.STRING },
                  geography: { type: Type.STRING },
                  timePeriod: { type: Type.STRING },
                  sampleSize: { type: Type.STRING },
                  sourceQuote: { type: Type.STRING },
                  section: { type: Type.STRING },
                },
                required: ['isReported', 'description'],
              },
              variables: {
                type: Type.OBJECT,
                properties: {
                  isReported: { type: Type.BOOLEAN },
                  dependent: {
                    type: Type.ARRAY,
                    items: {
                      type: Type.OBJECT,
                      properties: {
                        name: { type: Type.STRING },
                        proxy: { type: Type.STRING },
                        sourceQuote: { type: Type.STRING },
                      },
                      required: ['name'],
                    },
                  },
                  independent: {
                    type: Type.ARRAY,
                    items: {
                      type: Type.OBJECT,
                      properties: {
                        name: { type: Type.STRING },
                        proxy: { type: Type.STRING },
                        sourceQuote: { type: Type.STRING },
                      },
                      required: ['name'],
                    },
                  },
                  controls: {
                    type: Type.ARRAY,
                    items: {
                      type: Type.OBJECT,
                      properties: {
                        name: { type: Type.STRING },
                        proxy: { type: Type.STRING },
                        sourceQuote: { type: Type.STRING },
                      },
                      required: ['name'],
                    },
                  },
                },
                required: ['isReported', 'dependent', 'independent'],
              },
              keyFindings: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    finding: { type: Type.STRING },
                    significanceOrEffectSize: { type: Type.STRING },
                    sourceQuote: { type: Type.STRING },
                    section: { type: Type.STRING },
                  },
                  required: ['finding'],
                },
              },
              limitations: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    limitation: { type: Type.STRING },
                    sourceQuote: { type: Type.STRING },
                    section: { type: Type.STRING },
                    isReported: { type: Type.BOOLEAN },
                  },
                  required: ['limitation', 'isReported'],
                },
              },
              extractedFacts: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    category: { type: Type.STRING },
                    fact: { type: Type.STRING },
                    sourceQuote: { type: Type.STRING },
                    section: { type: Type.STRING },
                  },
                  required: ['category', 'fact', 'sourceQuote'],
                },
              },
              aiInterpretations: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    category: { type: Type.STRING },
                    interpretation: { type: Type.STRING },
                    groundedBasis: { type: Type.STRING },
                    confidence: { type: Type.STRING },
                  },
                  required: ['category', 'interpretation', 'groundedBasis'],
                },
              },
            },
            required: [
              'title',
              'authors',
              'year',
              'bfsiDomain',
              'researchObjective',
              'methodology',
              'sample',
              'variables',
              'keyFindings',
              'limitations',
              'extractedFacts',
              'aiInterpretations',
            ],
          },
        },
      });
    } else {
      // Plain text extraction
      response = await ai.models.generateContent({
        model: 'gemini-3.8-flash',
        contents: `${extractionInstruction}\n\n=== PAPER TEXT CONTENT ===\n${text}`,
        config: {
          systemInstruction: systemPrompt,
          responseMimeType: 'application/json',
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              title: { type: Type.STRING },
              authors: {
                type: Type.ARRAY,
                items: { type: Type.STRING },
              },
              year: { type: Type.STRING },
              journalOrSource: { type: Type.STRING },
              bfsiDomain: {
                type: Type.STRING,
                description: 'Must be one of: Banking, Finance, Insurance, FinTech, Cross-BFSI',
              },
              researchObjective: {
                type: Type.OBJECT,
                properties: {
                  value: { type: Type.STRING },
                  isReported: { type: Type.BOOLEAN },
                  sourceQuote: { type: Type.STRING },
                  section: { type: Type.STRING },
                },
                required: ['value', 'isReported'],
              },
              methodology: {
                type: Type.OBJECT,
                properties: {
                  value: { type: Type.STRING },
                  isReported: { type: Type.BOOLEAN },
                  sourceQuote: { type: Type.STRING },
                  section: { type: Type.STRING },
                },
                required: ['value', 'isReported'],
              },
              sample: {
                type: Type.OBJECT,
                properties: {
                  isReported: { type: Type.BOOLEAN },
                  description: { type: Type.STRING },
                  dataset: { type: Type.STRING },
                  geography: { type: Type.STRING },
                  timePeriod: { type: Type.STRING },
                  sampleSize: { type: Type.STRING },
                  sourceQuote: { type: Type.STRING },
                  section: { type: Type.STRING },
                },
                required: ['isReported', 'description'],
              },
              variables: {
                type: Type.OBJECT,
                properties: {
                  isReported: { type: Type.BOOLEAN },
                  dependent: {
                    type: Type.ARRAY,
                    items: {
                      type: Type.OBJECT,
                      properties: {
                        name: { type: Type.STRING },
                        proxy: { type: Type.STRING },
                        sourceQuote: { type: Type.STRING },
                      },
                      required: ['name'],
                    },
                  },
                  independent: {
                    type: Type.ARRAY,
                    items: {
                      type: Type.OBJECT,
                      properties: {
                        name: { type: Type.STRING },
                        proxy: { type: Type.STRING },
                        sourceQuote: { type: Type.STRING },
                      },
                      required: ['name'],
                    },
                  },
                  controls: {
                    type: Type.ARRAY,
                    items: {
                      type: Type.OBJECT,
                      properties: {
                        name: { type: Type.STRING },
                        proxy: { type: Type.STRING },
                        sourceQuote: { type: Type.STRING },
                      },
                      required: ['name'],
                    },
                  },
                },
                required: ['isReported', 'dependent', 'independent'],
              },
              keyFindings: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    finding: { type: Type.STRING },
                    significanceOrEffectSize: { type: Type.STRING },
                    sourceQuote: { type: Type.STRING },
                    section: { type: Type.STRING },
                  },
                  required: ['finding'],
                },
              },
              limitations: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    limitation: { type: Type.STRING },
                    sourceQuote: { type: Type.STRING },
                    section: { type: Type.STRING },
                    isReported: { type: Type.BOOLEAN },
                  },
                  required: ['limitation', 'isReported'],
                },
              },
              extractedFacts: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    category: { type: Type.STRING },
                    fact: { type: Type.STRING },
                    sourceQuote: { type: Type.STRING },
                    section: { type: Type.STRING },
                  },
                  required: ['category', 'fact', 'sourceQuote'],
                },
              },
              aiInterpretations: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    category: { type: Type.STRING },
                    interpretation: { type: Type.STRING },
                    groundedBasis: { type: Type.STRING },
                    confidence: { type: Type.STRING },
                  },
                  required: ['category', 'interpretation', 'groundedBasis'],
                },
              },
            },
            required: [
              'title',
              'authors',
              'year',
              'bfsiDomain',
              'researchObjective',
              'methodology',
              'sample',
              'variables',
              'keyFindings',
              'limitations',
              'extractedFacts',
              'aiInterpretations',
            ],
          },
        },
      });
    }

    const parsed = cleanAndParseJSON(response.text || '{}');
    return res.json({ success: true, data: parsed });
  } catch (error: any) {
    console.error('Error in /api/extract-paper:', error);
    return res.status(500).json({ error: error.message || 'Failed to extract paper data via Gemini.' });
  }
});

// 2. SYNTHESIZE LITERATURE MATRIX ROUTE
app.post('/api/synthesize-matrix', async (req, res) => {
  try {
    const { papers, userFocus } = req.body;

    if (!papers || !Array.isArray(papers) || papers.length === 0) {
      return res.status(400).json({ error: 'Please provide at least one paper to synthesize.' });
    }

    const systemPrompt = `You are an elite academic meta-analyst and journal editor in Banking, Finance, and Insurance (BFSI).
Your task is to analyze the supplied literature review matrix, identify thematic patterns, compare methodologies and sample distributions, and formulate provisional research questions.

CRITICAL CONSTRAINTS:
1. Every research gap MUST be explicitly tied to a limitation reported in the papers or an empirical divergence between them. Do NOT fabricate ungrounded gaps.
2. Clearly distinguish extracted facts (what the papers proved or reported) from AI-generated interpretations and proposals.
3. Formulate provisional research questions (PRQs) that are methodologically actionable, theoretical, and relevant to modern BFSI scholarship (e.g. Basel capital adequacy, climate risk, systemic risk, ESG pricing, algorithmic lending).
4. For each research question, specify:
   - The question text
   - Grounded rationale citing specific papers
   - Suggested theoretical foundation (e.g., Agency Theory, Buffer Theory of Capital, Asymmetric Information, Prospect Theory)
   - Candidate methodology (e.g., Difference-in-Differences, GMM panel, Event Study, Extreme Value Theory)
   - Candidate sample / data source`;

    const promptContent = `Analyze the following ${papers.length} academic papers extracted in the BFSI matrix:
${JSON.stringify(papers, null, 2)}

${userFocus ? `User research focus/priority: "${userFocus}"` : ''}

Synthesize these papers into:
1. Executive meta-synthesis summary across Banking, Finance, and Insurance
2. Thematic clusters (2 to 4 key themes connecting these papers)
3. Methodological and sample comparison matrix
4. Empirical consensus points vs. Tensions/Contradictions
5. Reported research gaps (only citing what the authors themselves reported or direct cross-paper omissions)
6. Provisional Research Questions (PRQs) formulated from this evidence (minimum 3 to 5 high-impact questions)`;

    const response = await ai.models.generateContent({
      model: 'gemini-3.8-flash',
      contents: promptContent,
      config: {
        systemInstruction: systemPrompt,
        responseMimeType: 'application/json',
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            executiveSummary: { type: Type.STRING },
            crossDomainSynthesis: {
              type: Type.OBJECT,
              properties: {
                bankingInsights: { type: Type.STRING },
                financeInsights: { type: Type.STRING },
                insuranceInsights: { type: Type.STRING },
              },
            },
            thematicClusters: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  themeTitle: { type: Type.STRING },
                  description: { type: Type.STRING },
                  coveredPaperTitles: {
                    type: Type.ARRAY,
                    items: { type: Type.STRING },
                  },
                  consensusFindings: {
                    type: Type.ARRAY,
                    items: { type: Type.STRING },
                  },
                  methodologicalDivergence: { type: Type.STRING },
                },
                required: ['themeTitle', 'description', 'coveredPaperTitles'],
              },
            },
            methodologicalComparison: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  paperTitle: { type: Type.STRING },
                  methodDesign: { type: Type.STRING },
                  sampleScope: { type: Type.STRING },
                  strengths: { type: Type.STRING },
                  vulnerabilities: { type: Type.STRING },
                },
                required: ['paperTitle', 'methodDesign', 'sampleScope'],
              },
            },
            empiricalTensions: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  issue: { type: Type.STRING },
                  divergentFindings: { type: Type.STRING },
                  contributingFactors: { type: Type.STRING },
                },
                required: ['issue', 'divergentFindings'],
              },
            },
            groundedGaps: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  gapDescription: { type: Type.STRING },
                  reportedByPaper: { type: Type.STRING },
                  sourceLimitationQuote: { type: Type.STRING },
                  researchOpportunity: { type: Type.STRING },
                },
                required: ['gapDescription', 'reportedByPaper'],
              },
            },
            provisionalResearchQuestions: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  id: { type: Type.STRING },
                  question: { type: Type.STRING },
                  targetDomain: { type: Type.STRING },
                  groundedRationale: { type: Type.STRING },
                  theoreticalFoundation: { type: Type.STRING },
                  suggestedMethodology: { type: Type.STRING },
                  candidateSampleOrData: { type: Type.STRING },
                  isAIGenerated: { type: Type.BOOLEAN },
                },
                required: [
                  'question',
                  'targetDomain',
                  'groundedRationale',
                  'theoreticalFoundation',
                  'suggestedMethodology',
                  'candidateSampleOrData',
                  'isAIGenerated',
                ],
              },
            },
          },
          required: [
            'executiveSummary',
            'thematicClusters',
            'methodologicalComparison',
            'empiricalTensions',
            'groundedGaps',
            'provisionalResearchQuestions',
          ],
        },
      },
    });

    const parsed = cleanAndParseJSON(response.text || '{}');
    return res.json({ success: true, data: parsed });
  } catch (error: any) {
    console.error('Error in /api/synthesize-matrix:', error);
    return res.status(500).json({ error: error.message || 'Failed to synthesize matrix.' });
  }
});

// 3. ASK GROUNDED QUESTION (Chat/Query over literature matrix)
app.post('/api/query-literature', async (req, res) => {
  try {
    const { query, papers } = req.body;

    if (!query) {
      return res.status(400).json({ error: 'Please provide a query.' });
    }

    const systemPrompt = `You are a scholarly literature assistant specializing in Banking, Finance, and Insurance.
Answer the researcher's query strictly based on the supplied literature matrix.
RULES:
1. Always cite the specific paper title and author.
2. Quote exact source text where available.
3. If an answer or variable is not reported in any paper, explicitly say "This information is not reported in the supplied papers."
4. Clearly distinguish extracted facts from any AI analytical interpretation.`;

    const response = await ai.models.generateContent({
      model: 'gemini-3.8-flash',
      contents: `RESEARCHER QUERY: "${query}"\n\nAVAILABLE PAPERS IN MATRIX:\n${JSON.stringify(papers, null, 2)}`,
      config: {
        systemInstruction: systemPrompt,
      },
    });

    return res.json({ success: true, answer: response.text });
  } catch (error: any) {
    console.error('Error in /api/query-literature:', error);
    return res.status(500).json({ error: error.message || 'Failed to answer literature query.' });
  }
});

// Start server and mount Vite
async function startServer() {
  if (!isProduction) {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    // Serve static files in production
    app.use(express.static(path.resolve(__dirname, 'dist')));
    app.get('*', (_req, res) => {
      res.sendFile(path.resolve(__dirname, 'dist', 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`BFSI Research Assistant server running at http://0.0.0.0:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error('Fatal error starting server:', err);
  process.exit(1);
});
