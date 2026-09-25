using System.Diagnostics;
using Jevlet.Core;
using Xunit.Abstractions;

namespace Jevlet.Core.Tests;

public class LatencyTests(ITestOutputHelper output)
{
    [SkippableTheory]
    [InlineData(true)]
    [InlineData(false)]
    public void Skill_and_risk_pass_is_interactive(bool preferGpu)
    {
        var model = GoldenTests.ModelDirectory();
        Skip.If(model is null, "no exported model");
        using var engine = DecisionEngine.Load(model!, preferGpu);
        var catalogue = Catalogue.Shared;
        var questions = new List<Question>
        {
            new(catalogue.SkillQuestion, catalogue.Skills.Select(s => s.Name).ToList()),
            new(catalogue.RiskQuestion, ["True", "False"], "noul"),
        };
        var state = catalogue.State("remind me to call the bank tomorrow at 5", "OUTLOOK: Inbox - Outlook");
        engine.Evaluate(state, questions); // warm-up
        var timings = new List<double>();
        for (var run = 0; run < 20; run++)
        {
            var watch = Stopwatch.StartNew();
            var answers = engine.Evaluate(state, questions);
            timings.Add(watch.Elapsed.TotalMilliseconds);
            Assert.Equal(2, answers.Count);
        }
        timings.Sort();
        var median = timings[timings.Count / 2];
        output.WriteLine($"{engine.Device}: median {median:F1} ms, p90 {timings[(int)(timings.Count * 0.9)]:F1} ms, tokens {engine.Packer.Pack(state, questions).InputIds.Length}");
        Assert.True(median < 400, $"median {median} ms");
    }
}
