import {
  Button,
  Callout,
  Card,
  CardBody,
  CardHeader,
  Divider,
  Grid,
  H1,
  H2,
  H3,
  Pill,
  Row,
  Stack,
  Stat,
  Table,
  Text,
  useCanvasAction,
  useHostTheme,
} from "cursor/canvas";

const techniques = [
  [
    "1",
    "Representation + auxiliary learning",
    "Shared encoder learns policy, win probability, prize differential, and legal-count prediction",
    "Tested",
    "E1 outcome/count heads: three nulls at n=2,000",
  ],
  [
    "2",
    "Matchup-conditioned mixture",
    "Small observable-state router selects policy adapters or rule masks",
    "Tested",
    "E2 adapters: mirror screen null; Alakazam cell worse than control",
  ],
  [
    "3",
    "DAgger with a strong teacher",
    "Collect states visited by the clone; label only uncertain/high-impact decisions with planner or human review",
    "Medium",
    "Lower on-policy error plus positive arena delta",
  ],
  [
    "4",
    "Conservative offline RL",
    "Use BC as an anchor; learn value/Q heads from outcomes and move only supported actions",
    "Medium",
    "First pass off-policy checks, then ≥0.541 vs control",
  ],
  [
    "5",
    "BC-guided planning and distillation",
    "Use policy top-k as the prior, value head at leaves, then distill improved actions back into the fast policy",
    "Round 2",
    "Time-safe and positive at n=200 before full run",
  ],
  [
    "6",
    "Population self-play / PSRO",
    "Train against a weighted population of decks and policy snapshots rather than mirror self-play",
    "Later",
    "Improves worst-matchup and weighted win rate",
  ],
];

const experiments = [
  ["E1", "Outcome + count auxiliary heads", "Settled null", "0.505 / 0.507 / 0.500 vs seed-matched control"],
  ["E2", "Train two matchup adapters with an observable-state router", "Settled null", "Mirror 0.521 [0.490, 0.552]; Alakazam treatment 0.782 vs control 0.792"],
  ["E3", "Review 160 uncertain live-trajectory decisions", "Teacher blocked", "15-label pilot is audit-only; no qualified teacher available"],
  ["E4", "Re-run outcome learning with conservative support constraints", "Blocked", "E1/E3 prerequisites failed"],
  ["E5", "Round-2 BC-prior beam search; distill wins", "High", "Converts extra compute into a deployable fast policy"],
];

function OpenFileButton({ path, label }: { path: string; label: string }) {
  const dispatch = useCanvasAction();
  return (
    <Button variant="secondary" onClick={() => dispatch({ type: "openFile", path })}>
      {label}
    </Button>
  );
}

function ArchitectureCard({
  title,
  status,
  children,
}: {
  title: string;
  status: string;
  children: string;
}) {
  return (
    <Card>
      <CardHeader trailing={<Pill size="sm">{status}</Pill>}>{title}</CardHeader>
      <CardBody>
        <Text>{children}</Text>
      </CardBody>
    </Card>
  );
}

export default function BeyondBehavioralCloning() {
  const theme = useHostTheme();

  return (
    <Stack gap={20} style={{ padding: 24, maxWidth: 1180, margin: "0 auto" }}>
      <Stack gap={8}>
        <Row align="center" justify="space-between" wrap>
          <H1>Beyond behavioral cloning</H1>
          <Pill active>Evidence-backed roadmap</Pill>
        </Row>
        <Text tone="secondary">
          Keep the clone as a strong prior. Add outcome prediction, context routing,
          interactive data collection, and budgeted planning around it.
        </Text>
      </Stack>

      <Callout tone="info" title="Recommended direction">
        Do not replace BC with end-to-end RL. Build a hybrid policy: BC for safe action
        priors, hard masks for genuinely dominated moves, learned value heads for
        outcomes, matchup adapters for flexibility, and search only where uncertainty
        and available compute justify it.
      </Callout>

      <Grid columns={4} gap={12}>
        <Stat value="2,810" label="Human games cloned" />
        <Stat value="95.6%" label="Encoding ceiling measured" tone="success" />
        <Stat value="~69–72%" label="Clone top-1 agreement" tone="warning" />
        <Stat value="E1+E2 null" label="Beyond-BC screens so far" tone="danger" />
      </Grid>

      <Divider />

      <Stack gap={12}>
        <H2>Target architecture</H2>
        <Grid columns="repeat(4, minmax(0, 1fr))" gap={12}>
          <ArchitectureCard title="1. BC prior" status="Keep">
            Listwise policy head supplies fast, legal action rankings and protects
            against unsupported exploration.
          </ArchitectureCard>
          <ArchitectureCard title="2. Safety masks" status="Proven">
            Arithmetic rules remove strictly dominated choices. Tradeoff decisions stay
            learned and context dependent.
          </ArchitectureCard>
          <ArchitectureCard title="3. Multi-task learner" status="E1/E2 null">
            Outcome/count auxiliaries and hard-routed matchup adapters both learned
            diagnostics without clearing strength screens. Value learning stays blocked.
          </ArchitectureCard>
          <ArchitectureCard title="4. Planner / teacher" status="Conditional">
            BC narrows the branch factor; a value head scores leaves. Improved actions
            are distilled back into the policy.
          </ArchitectureCard>
        </Grid>
        <div
          style={{
            borderLeft: `3px solid ${theme.accent.primary}`,
            paddingLeft: 14,
            color: theme.text.secondary,
          }}
        >
          The key change is architectural separation: imitation proposes, rules constrain,
          value learning evaluates, and context decides how much computation to spend.
        </div>
      </Stack>

      <Grid columns="1.45fr 1fr" gap={18} align="start">
        <Stack gap={10}>
          <H2>Technique portfolio</H2>
          <Table
            headers={["Order", "Technique", "Role", "Priority", "Gate"]}
            rows={techniques}
            columnAlign={["center", "left", "left", "center", "left"]}
            rowTone={["success", "success", "info", "warning", "info", "neutral"]}
            striped
          />
        </Stack>

        <Stack gap={12}>
          <H2>Why this ordering</H2>
          <Card variant="borderless">
            <CardBody style={{ padding: 0 }}>
              <Stack gap={12}>
                <Stack gap={4}>
                  <H3>Features paid; capacity did not</H3>
                  <Text tone="secondary">
                    Three feature generations gained roughly +115, +37, then +14 Elo.
                    An 8.2× larger network barely changed decisions.
                  </Text>
                </Stack>
                <Stack gap={4}>
                  <H3>Naive outcome reweighting was not enough</H3>
                  <Text tone="secondary">
                    AWR-style self-play was null at 4k and 16k games. The next attempt
                    needs support constraints, a learned critic, and population diversity.
                  </Text>
                </Stack>
                <Stack gap={4}>
                  <H3>Search needs a better contract</H3>
                  <Text tone="secondary">
                    Existing search lost to BC under first-round constraints. Revisit it
                    only as a BC-prior teacher on Round-2 hardware, with strict time gates.
                  </Text>
                </Stack>
              </Stack>
            </CardBody>
          </Card>
          <Callout tone="warning" title="Avoid repeating measured dead ends">
            More undifferentiated replay data, a larger MLP, global tradeoff rules, and
            unconditioned mirror self-play are unlikely to improve this agent.
          </Callout>
        </Stack>
      </Grid>

      <Divider />

      <Stack gap={10}>
        <H2>Five-experiment implementation sequence</H2>
        <Table
          headers={["Step", "Experiment", "Cost", "What it resolves"]}
          rows={experiments}
          columnAlign={["center", "left", "center", "left"]}
          rowTone={["success", "neutral", "info", "warning", "info"]}
        />
        <Text size="small" tone="tertiary">
          Every candidate should be compared against a byte-identical or seed-matched
          control. Validation accuracy is diagnostic; multi-anchor arena win rate is the
          shipping criterion. E1 and E2 are settled nulls; E4 stays blocked.
        </Text>
      </Stack>

      <Row gap={8} wrap>
        <OpenFileButton path="agents/sa/bcagent.py" label="Runtime policy" />
        <OpenFileButton path="scripts/train_policy.py" label="Training pipeline" />
        <OpenFileButton path="scripts/p26_selfplay_gen.py" label="Outcome data generator" />
        <OpenFileButton path="ROADMAP.md" label="Measured evidence" />
      </Row>
    </Stack>
  );
}
