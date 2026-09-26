using System.Diagnostics;
using Jevlet.Core;
using Xunit.Abstractions;

namespace Jevlet.Core.Tests;

/// <summary>
/// Latency after an idle pause, per GPU adapter: typing leaves the GPU idle between plans, and a
/// low-power GPU pays a wake-up cost a tight benchmark loop never sees. Run on demand:
/// JEVLET_IDLE_TEST=1 dotnet test --filter IdleLatency
/// </summary>
public class IdleLatencyTests(ITestOutputHelper output)
{
    [SkippableTheory]
    [InlineData(0)]
    [InlineData(1)]
    public void Reports_latency_after_idle(int adapter)
    {
        Skip.If(Environment.GetEnvironmentVariable("JEVLET_IDLE_TEST") != "1", "on demand only");
        var model = GoldenTests.ModelDirectory();
        Skip.If(model is null, "no exported model");
        DecisionEngine engine;
        try
        {
            engine = DecisionEngine.Load(model!, preferGpu: true, gpuDevice: adapter);
        }
        catch (Exception error)
        {
            Skip.If(true, $"adapter {adapter}: {error.Message}");
            return;
        }
        using var _ = engine;
        Skip.If(engine.Device != "gpu", $"adapter {adapter} unavailable");
        var catalogue = Catalogue.Shared;
        var questions = new List<Question>
        {
            new(catalogue.SkillQuestion, catalogue.Skills.Select(s => s.Name).ToList()),
            new(catalogue.RiskQuestion, ["True", "False"], "noul"),
        };
        var state = catalogue.State("what's 18 percent of 240", "none");
        engine.Evaluate(state, questions);
        foreach (var idle in new[] { 0, 500, 2000 })
        {
            var timings = new List<double>();
            for (var run = 0; run < 6; run++)
            {
                Thread.Sleep(idle);
                var watch = Stopwatch.StartNew();
                engine.Evaluate(state, questions);
                timings.Add(watch.Elapsed.TotalMilliseconds);
            }
            timings.Sort();
            output.WriteLine($"adapter {adapter}, idle {idle} ms: median {timings[3]:F1} ms, max {timings[^1]:F1} ms");
        }
    }
}
