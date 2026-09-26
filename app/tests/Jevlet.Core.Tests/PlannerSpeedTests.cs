using System.Diagnostics;
using Jevlet.Core;
using Xunit.Abstractions;

namespace Jevlet.Core.Tests;

/// <summary>Where planning time goes, stage by stage, on a realistic desktop (300+ apps).</summary>
public class PlannerSpeedTests(ITestOutputHelper output)
{
    private static readonly List<string> Apps =Enumerable.Range(0, 333)
        .Select(i => $"{new[] { "Microsoft", "Adobe", "Google", "Visual", "Windows", "NVIDIA", "Intel", "Zoom" }[i % 8]} {new[] { "Photo Editor", "Studio Code", "Media Player", "Terminal", "Calculator", "Control Center", "Update Assistant" }[i % 7]} {i}")
        .ToList();

    private static double Time(Action action, int runs = 20)
    {
        action();
        var watch = Stopwatch.StartNew();
        for (var run = 0; run < runs; run++)
        {
            action();
        }
        return watch.Elapsed.TotalMilliseconds / runs;
    }

    [Fact]
    public void Text_rules_are_fast_enough_to_run_on_every_keystroke()
    {
        var shortlist = Time(() => TextRules.Shortlist("open calculator", Apps, 8));
        var spans = Time(() => TextRules.SpanCandidates("remind me to call the bank tomorrow at 5 please"));
        var time = Time(() => TimeParser.Parse("remind me to call the bank tomorrow at 5", DateTime.Now));
        output.WriteLine($"shortlist over {Apps.Count} apps: {shortlist:F2} ms; spans: {spans:F2} ms; time: {time:F3} ms");
        Assert.True(shortlist < 10, $"shortlist {shortlist} ms");
        Assert.True(spans < 5, $"spans {spans} ms");
    }
}
