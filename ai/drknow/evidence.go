package drknow

/*
Evidence is a way to strengthen a claim.
*/
type Evidence struct {
	Description string
	Data        interface{}
	Weight      float64
}
