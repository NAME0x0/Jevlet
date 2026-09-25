using System.Text.Json;
using Jevlet.Core;

namespace Jevlet.Core.Tests;

/// <summary>
/// Replays outputs recorded from the Python implementation (scripts/export_onnx.py and
/// scripts/make_text_goldens.py). Any divergence means the app would offer the model
/// different options, or different tokens, than training did.
/// </summary>
public class GoldenTests
{
    private static readonly string Golden = Path.Combine(AppContext.BaseDirectory, "golden");
    private static readonly WordPieceTokenizer Tokenizer = WordPieceTokenizer.FromFile(Path.Combine(Golden, "vocab.txt"));

    private static JsonElement Read(string name) =>
        JsonDocument.Parse(File.ReadAllText(Path.Combine(Golden, name))).RootElement;

    private static List<string> Strings(JsonElement array) => array.EnumerateArray().Select(v => v.GetString()!).ToList();

    [Fact]
    public void Tokenizer_matches_hugging_face_ids()
    {
        var failures = new List<string>();
        foreach (var row in Read("tokenizer.json").EnumerateArray())
        {
            var text = row.GetProperty("text").GetString()!;
            var expected = row.GetProperty("ids").EnumerateArray().Select(v => v.GetInt32()).ToList();
            var actual = Tokenizer.Encode(text);
            if (!expected.SequenceEqual(actual))
            {
                failures.Add($"{text}: expected [{string.Join(',', expected)}] got [{string.Join(',', actual)}]");
            }
        }
        Assert.Empty(failures);
    }

    [Fact]
    public void Packer_builds_the_same_rows_as_the_python_collator()
    {
        var golden = Read("packing.json");
        var packer = new Packer(Tokenizer, ModelManifest.Parse(golden.GetProperty("model")));
        var failures = new List<string>();
        foreach (var row in golden.GetProperty("cases").EnumerateArray())
        {
            var packed = packer.Pack(row.GetProperty("state").GetString()!, Questions(row));
            var id = row.GetProperty("id").GetString();
            var ids = row.GetProperty("input_ids").EnumerateArray().Select(v => v.GetInt64());
            var positions = row.GetProperty("position_ids").EnumerateArray().Select(v => v.GetInt64());
            if (!ids.SequenceEqual(packed.InputIds))
            {
                failures.Add($"{id}: input ids differ");
            }
            if (!positions.SequenceEqual(packed.PositionIds))
            {
                failures.Add($"{id}: positions differ");
            }
            var records = row.GetProperty("records").EnumerateArray().ToList();
            for (var q = 0; q < records.Count; q++)
            {
                var spans = records[q].GetProperty("option_spans").EnumerateArray()
                    .Select(s => (s[0].GetInt32(), s[1].GetInt32()));
                if (!spans.SequenceEqual(packed.Questions[q].OptionSpans)
                    || records[q].GetProperty("decide_position").GetInt32() != packed.Questions[q].DecidePosition)
                {
                    failures.Add($"{id}: question {q} spans or decide position differ");
                }
            }
        }
        Assert.Empty(failures);
    }

    [SkippableFact]
    public void Onnx_logits_match_pytorch()
    {
        var model = ModelDirectory();
        Skip.If(model is null, "no exported model (run scripts.export_onnx)");
        var golden = Read("packing.json");
        using var engine = DecisionEngine.Load(model!);
        Skip.IfNot(engine.Manifest.SourceCheckpoint == golden.GetProperty("model").GetProperty("source_checkpoint").GetString(),
            "exported model differs from the golden's");
        var worst = 0.0;
        foreach (var row in golden.GetProperty("cases").EnumerateArray())
        {
            var logits = engine.Logits(engine.Packer.Pack(row.GetProperty("state").GetString()!, Questions(row)));
            var expected = row.GetProperty("logits").EnumerateArray().ToList();
            for (var q = 0; q < expected.Count; q++)
            {
                var values = expected[q].EnumerateArray().Select(v => v.GetDouble()).ToList();
                for (var o = 0; o < values.Count; o++)
                {
                    worst = Math.Max(worst, Math.Abs(values[o] - logits[q][o]));
                }
            }
        }
        Assert.True(worst < 2e-3, $"max logit difference {worst}");
    }

    [Fact]
    public void Text_rules_match_python()
    {
        var golden = Read("text.json");
        var apps = Strings(golden.GetProperty("apps"));
        var failures = new List<string>();
        foreach (var row in golden.GetProperty("cases").EnumerateArray())
        {
            var command = row.GetProperty("command").GetString()!;
            void Check(string what, bool same)
            {
                if (!same)
                {
                    failures.Add($"{what}: {command}");
                }
            }
            Check("courtesy", TextRules.StripCourtesy(command) == row.GetProperty("courtesy").GetString());
            var spans = TextRules.SpanCandidates(command);
            var expectedSpans = Strings(row.GetProperty("spans"));
            Check($"spans [{string.Join(" | ", expectedSpans)}] vs [{string.Join(" | ", spans)}]", expectedSpans.SequenceEqual(spans));
            Check("has_time", TextRules.HasTime(command) == row.GetProperty("has_time").GetBoolean());
            var duration = row.GetProperty("duration");
            int? expectedDuration = duration.ValueKind == JsonValueKind.Null ? null : duration.GetInt32();
            Check("duration", TextRules.ParseDuration(command) == expectedDuration);
            Check("shortlist", Strings(row.GetProperty("shortlist")).SequenceEqual(TextRules.Shortlist(command, apps, 8)));
            var top = TextRules.Shortlist(command, apps, 1)[0];
            Check("similarity", Math.Abs(TextRules.Similarity(command, top) - row.GetProperty("top_similarity").GetDouble()) < 1e-9);
        }
        Assert.True(failures.Count == 0, $"{failures.Count} differences:\n{string.Join('\n', failures.Take(25))}");
    }

    [Fact]
    public void Slot_options_match_python()
    {
        var golden = Read("slots.json");
        var env = golden.GetProperty("environment");
        var snapshot = new Snapshot(
            Strings(env.GetProperty("apps")), Strings(env.GetProperty("windows")),
            env.GetProperty("current_window").GetString()!, Strings(env.GetProperty("events")),
            Strings(env.GetProperty("alarms")), Strings(env.GetProperty("todos")),
            Strings(env.GetProperty("files")), Strings(env.GetProperty("reminders")));
        var failures = new List<string>();
        foreach (var row in golden.GetProperty("cases").EnumerateArray())
        {
            var command = row.GetProperty("command").GetString()!;
            var slot = row.GetProperty("slot").GetString()!;
            if (!Strings(row.GetProperty("options")).SequenceEqual(SlotOptions.For(slot, command, snapshot)))
            {
                failures.Add($"{slot}: {command}");
            }
        }
        Assert.True(failures.Count == 0, $"{failures.Count} differences:\n{string.Join('\n', failures.Take(25))}");
    }

    [Fact]
    public void Catalogue_has_fifty_skills_and_every_slot_has_a_question()
    {
        var catalogue = Catalogue.Shared;
        Assert.Equal(50, catalogue.Skills.Count);
        foreach (var skill in catalogue.Skills)
        {
            foreach (var slot in skill.Slots.Concat(skill.OptionalSlots))
            {
                Assert.True(catalogue.SlotQuestions.ContainsKey(slot), $"{skill.Key}.{slot}");
            }
        }
    }

    private static List<Question> Questions(JsonElement row) =>
        row.GetProperty("questions").EnumerateArray()
            .Select(q => new Question(q.GetProperty("text").GetString()!, Strings(q.GetProperty("options")), q.GetProperty("kind").GetString()!))
            .ToList();

    internal static string? ModelDirectory()
    {
        var configured = Environment.GetEnvironmentVariable("JEVLET_MODEL_DIR");
        if (!string.IsNullOrEmpty(configured))
        {
            return configured;
        }
        for (var directory = new DirectoryInfo(AppContext.BaseDirectory); directory is not null; directory = directory.Parent)
        {
            var candidate = Path.Combine(directory.FullName, "data", "models", "onnx", "v4");
            if (File.Exists(Path.Combine(candidate, "model.onnx")))
            {
                return candidate;
            }
        }
        return null;
    }
}
