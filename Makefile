.PHONY: all run capnp

run:
	docker compose down
	docker build -t caramba .
	docker compose up -d --build
	go run main.go example --user $(USER)