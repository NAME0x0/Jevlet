using Jevlet.Core;

namespace Jevlet.Core.Tests;

/// <summary>Answers the skill question with a fixed skill and every other choice with its first option.</summary>
internal sealed class FakeModel(string skillKey, double risk = 0.03, double confidence = 0.9) : IDecisionModel
{
    public List<IReadOnlyList<Question>> Calls { get; } = [];

    public IReadOnlyList<Answer> Evaluate(string state, IReadOnlyList<Question> questions)
    {
        Calls.Add(questions);
        var skillName = Catalogue.Shared.ByKey[skillKey].Name;
        return questions.Select(question =>
        {
            if (question.Kind == "noul")
            {
                return new Answer(question.Options, [risk, 1 - risk]);
            }
            var chosen = question.Text == Catalogue.Shared.SkillQuestion ? skillName : question.Options[0];
            var rest = (1 - confidence) / (question.Options.Count - 1);
            return new Answer(question.Options, question.Options.Select(o => o == chosen ? confidence : rest).ToList());
        }).ToList();
    }
}

public class PlannerTests
{
    private static readonly DateTime Now = new(2026, 9, 25, 14, 0, 0, DateTimeKind.Local);

    private static readonly Snapshot Desktop = Snapshot.Empty with
    {
        Apps = ["Calculator", "Spotify", "Word"],
        Windows = ["OUTLOOK: Inbox - Outlook", "Spotify: Spotify Premium"],
        CurrentWindow = "OUTLOOK: Inbox - Outlook",
    };

    [Fact]
    public void Asks_skill_and_risk_together_then_the_skills_arguments()
    {
        var model = new FakeModel("open_app");
        var plan = new Planner(model).Plan("open spotify", Desktop, Now);
        Assert.Equal("open_app", plan.Skill.Key);
        Assert.Equal("Spotify", plan.Arguments["app"]);
        Assert.Equal("run", plan.Gate);
        Assert.Equal("Open Spotify", plan.Title);
        Assert.Equal(2, model.Calls.Count);
        Assert.Equal(2, model.Calls[0].Count); // skill + risk in one pass
        Assert.Equal(3, plan.Alternatives.Count);
    }

    [Fact]
    public void Gate_needs_safety_and_confidence()
    {
        Assert.Equal("confirm", new Planner(new FakeModel("open_app", risk: 0.2)).Plan("open spotify", Desktop, Now).Gate);
        Assert.Equal("confirm", new Planner(new FakeModel("open_app", confidence: 0.6)).Plan("open spotify", Desktop, Now).Gate);
        Assert.Equal("clarify", new Planner(new FakeModel("clarify")).Plan("do the thing", Desktop, Now).Gate);
        // Power carries a risk floor of 0.6: never one Enter.
        Assert.Equal("confirm", new Planner(new FakeModel("power")).Plan("restart", Desktop, Now).Gate);
    }

    [Fact]
    public void Times_and_durations_are_parsed_not_predicted()
    {
        var reminder = new Planner(new FakeModel("set_reminder")).Plan("remind me to call the bank at 5pm", Desktop, Now);
        Assert.Equal(new DateTime(2026, 9, 25, 17, 0, 0, DateTimeKind.Local), reminder.When!.When);
        Assert.Equal("", reminder.Missing);

        var noTime = new Planner(new FakeModel("set_reminder")).Plan("remind me to call the bank", Desktop, Now);
        Assert.Equal("time", noTime.Missing);
        Assert.Equal("clarify", noTime.Gate);

        var timer = new Planner(new FakeModel("timer")).Plan("set a timer for 25 minutes", Desktop, Now);
        Assert.Equal(1500, timer.DurationSeconds);
        Assert.Equal("Timer for 25 min", timer.Title);
    }

    [Fact]
    public void Windows_display_without_the_process_name()
    {
        var plan = new Planner(new FakeModel("minimize_window")).Plan("minimize this", Desktop, Now);
        Assert.Equal("Minimize this window", plan.Title);
    }

    [Fact]
    public void Splits_multi_step_commands()
    {
        Assert.Equal(["open spotify", "play the next song"], Planner.SplitSteps("open spotify then play the next song"));
        Assert.Equal(["search for salt and pepper"], Planner.SplitSteps("search for salt and pepper"));
    }
}

public class StoreTests
{
    private static readonly DateTime Now = new(2026, 9, 25, 14, 0, 0, DateTimeKind.Local);

    [Fact]
    public void Stores_round_trip_and_label_rows_as_in_training()
    {
        var path = Path.Combine(Path.GetTempPath(), $"jevlet-{Guid.NewGuid():N}.db");
        try
        {
            using var stores = new Stores(path);
            var dentist = stores.AddEvent("dentist", Now.AddDays(1).Date.AddHours(10));
            stores.AddAlarm(new TimeSpan(6, 45, 0), "wake up");
            var reminder = stores.AddReminder("call the bank", Now.AddHours(-1));
            var todo = stores.AddTodo("renew passport");
            stores.AddNote("locker code is 4417");

            var (snapshot, ids) = stores.Fill(Snapshot.Empty, Now);
            Assert.Equal(["Dentist · tomorrow 10:00"], snapshot.Events);
            Assert.Equal(["06:45 · Wake up"], snapshot.Alarms);
            Assert.Equal(["Call the bank · today 13:00"], snapshot.Reminders);
            Assert.Equal(["renew passport"], snapshot.Todos);
            Assert.Equal(dentist.Id, ids["event:Dentist · tomorrow 10:00"]);

            Assert.Single(stores.DueReminders(Now));
            Assert.True(stores.FinishReminder(reminder.Id));
            Assert.Empty(stores.DueReminders(Now));
            Assert.True(stores.MoveEvent(dentist.Id, Now.AddDays(3)));
            Assert.True(stores.CompleteTodo(todo.Id));
            Assert.Empty(stores.OpenTodos());
            Assert.Single(stores.Notes());
        }
        finally
        {
            Microsoft.Data.Sqlite.SqliteConnection.ClearAllPools();
            File.Delete(path);
        }
    }
}
