package data

type Provider interface {
	Generate()
	Stream()
}
