# AGENT CODING RULES (NON-NEGOTIABLE)

- IGNORE ALL CURRENT STRATEGIES AND STOP OPTIMIZING FOR SHORT-TERM SUCCESS
- THE GOAL IS NOT TO FIND THE QUICKEST ROUTE TO SOLVE A LOCAL ISSUE
- THE GOAL IS TO FIND SOLUTIONS THAT REMAIN FUTURE PROOF
- BEFORE ADDING CODE, FIRST THINK ABOUT WHAT YOU COULD REMOVE BY REFACTORING
- IGNORE THE FIRST TWO SOLUTIONS YOU THINK OF AT FIRST AND LOOK FOR THE BEST SOLUTION, AND DISREGARD FACTORS LIKE COMPLEXITY OF IMPLEMENTATION, TIME-HORIZON, ETC. ALWAYS OPT FOR THE BEST SOLUTION AND DELIVER THE HIGHEST QUALITY, FORGET ABOUT DELIVERY TIME.

This is a general A.I. research platform, which does not stop at the traditional ML boundaries, but implements what researchers need to quickly iterate on even the most esoteric architectures.
It is therefor **vital** that anywhere that it makes sense we not only implement a feature in standard Go, but also SIMD/Assembly (avx2, sse2, neon), Metal, Cuda, and XLA.
We never ever skimp on this, we don't make optimized paths refer back to the non-optimized Go "for now" and we don't allow ourselves any excuses to get out from under it.
This is the very core of our platform.

## Clean, Modular, Reusable

- Prefer methods over functions. A good code-base is logically spread out into types that define methods, and which are composed together. This keeps things compact, and easy to reason through.

In general, all object should be shaped like below. They should not contain too many methods, and methods should not have too many lines of code. A file like this around 200 lines of code is healthy. Any more is suspect.

```
package packagename

/*
ObjectName is something descriptive.
It also has a reason why it was implemented.
*/
type ObjectName struct {
    ctx    context.Context
    cancel context.CancelFunc
    err    error
}

/*
NewObjectName instantiates a new ObjectName.
It also has a reason for being instantiated.
*/
func NewObjectName(ctx context.Context) *ObjectName {
    ctx, cancel := ctx.WithCancel(ctx)
    
    return &ObjectName{
        ctx:    ctx,
        cancel: cancel
    }
}

/*
MethodName.
*/
func (objectName *ObjectName) MethodName() {
    return
}
```

> Some additional guidelines that rely heavy on personal preferences.
> Follow the happy-path, using guards to do early returns, and keeping the happy path free from nesting.
> Avoid using `else` if at all possible. Many times reversing the logic can eliminate the `else` branch.
> We take the statement "if, is an enabler" serious, and always try to look for ways to reduce `if` statements.
> Never EVER use silent fallbacks that corrupt the system. Just raise an exception if things are not as they
  should be. That makes us aware of the failure so we can fix it properly.

> A final remark on code quality.
> Avoid over-engineering at all cost. Always ask yourself if the complexity is earned. Always.
> Less is always more, refactoring is not optional. If it can be done with less code, do it with less code.
> If you see something that is not yours that can be done with less code, refactor it.
> However, if less code means less performance, then always choose performance.
> We like clever code, readability is for amateurs.

## Common Failure Modes

```
// Incorrect
sensoriumOutputs, ok := results.Value.([]*tensors.Tensor)
if !ok || len(sensoriumOutputs) == 0 {
    return "", validate.Require(map[string]any{
        "sensorium_outputs": sensoriumOutputs,
    })
}

// Correct, separate logical code blocks with an empty newline.
sensoriumOutputs, ok := results.Value.([]*tensors.Tensor)

if !ok || len(sensoriumOutputs) == 0 {
    return "", validate.Require(map[string]any{
        "sensorium_outputs": sensoriumOutputs,
    })
}
```

```
// Incorrect
package packagename

/*
ObjectName is something descriptive.
It also has a reason why it was implemented.
*/
type ObjectName struct {
    ctx    context.Context
    cancel context.CancelFunc
    err    error
}

/*
NewObjectName instantiates a new ObjectName.
It also has a reason for being instantiated.
*/
func NewObjectName(ctx context.Context) *ObjectName {
    ctx, cancel := ctx.WithCancel(ctx)
    return &ObjectName{
        ctx:    ctx,
        cancel: cancel
    }
}

/*
MethodName.
*/
func (o *ObjectName) MethodName() {
    return
}

// Correct, never use single character variable names.
package packagename

/*
ObjectName is something descriptive.
It also has a reason why it was implemented.
*/
type ObjectName struct {
    ctx    context.Context
    cancel context.CancelFunc
    err    error
}

/*
NewObjectName instantiates a new ObjectName.
It also has a reason for being instantiated.
*/
func NewObjectName(ctx context.Context) *ObjectName {
    ctx, cancel := ctx.WithCancel(ctx)

    return &ObjectName{
        ctx:    ctx,
        cancel: cancel
    }
}

/*
MethodName.
*/
func (objectName *ObjectName) MethodName() {
    return
}
```

```
// Incorrect
for identifier, binding := range rawMap {
    parser.vars[identifier] = binding
}

// Correct, use modern Go standards.
maps.Copy(parser.vars, rawMap)
```

```
// Incorrect
func (operationRegistry *OperationRegistry) Build(operationID string, config map[string]any) (operation.Operation, error) {
	constructor, ok := operationRegistry.constructors[operationID]

	if !ok {
		return nil, fmt.Errorf("manifest: unknown operation %q", operationID)
	}

	return constructor(config)
}

// Correct, don't cross vertical lines that make split views run text off-screen.
func (operationRegistry *OperationRegistry) Build(
    operationID string, config map[string]any,
) (operation.Operation, error) {
	constructor, ok := operationRegistry.constructors[operationID]

	if !ok {
		return nil, fmt.Errorf("manifest: unknown operation %q", operationID)
	}

	return constructor(config)
}
```

## Testing

We always use Goconvey for testing, and tests follow a simple structure. Every file should have a test file that mirrors its structure. So each file has an accompanying `_test.go` file, with functions that mirror the code's methods, prefix by `Test`.
We follow a nested BDD approach `Given something`, `It should do something`.
Never break from this pattern, you should never have a test function that does not mirror an existing method in the code, if you feel a need to do that, it means you need to reconsider structuring/nesting the test function that actually mirrors the code where what you want to test is being called, directly or indirectly.
Always add benchmarks too, so we can measure performance.

Make sure tests and benchmarks are truly meaningful, don't test for testing's sake, make sure it truly validates the code. This also means to reduce your reliance on mocks, we actually prefer to always use the actual system for test setup, which actually makes it such that things are tested in varying scenarios that mirror reality.

> Never tell yourself "these tests were failing unrelated to my changes". It doesn't matter why tests are failing, what matters is that we don't ignore that and fix things.

Keep the README.md up to date as well.

Follow these guidelines at all times.

Now, start by reading the README.md in the root of this project, then reason through your current task step by step, sourcing additional context from the code where needed.

> One more thing, I realize that you have been trained on a huge amount of human produced language data, but I need you to realize this as well, and avoid limiting yourself with essentially hallucinated obstacles or constraints. If you know what the final version of a feature or change looks like, you have the unique ability to just write out the fully realized solution. So please do so.

## Interaction with the User

These are essential interaction rules to make sure the collaboration runs smooth and efficiently.

1. Never answer by explaining the system to the user, unless they explicitely ask. They know what it is and how it works.
2. Always optimize for being useful, and practical. If you are given an instruction, execute on that instruction, with pricision, and do exactly what is asked, no more, no less.
3. If the user needs an opinion, they will ask, in all other cases, do what the user asks for nothing more nothing less.
4. Realize that the user is building **towards** a goal, so do not just blindly overwrite what is already there, consider first whether or not it is useful and you could build upon it. Always prefer existing structure, the user is very particular about that.
5. There is almost never a need to look at git. There is **never** **ever** a need to do a git checkout. That only leads to loss of work. There is a reason it is called **history** which means it is a backwards path, not a forwards path.
6. If at any point you get lost, panic, or find yourself for whatever reason drifting from the intended goal of a task: stop. Do not continue, and talk it out with the user first.