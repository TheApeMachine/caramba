package main

import (
	"fmt"
	"github.com/theapemachine/caramba/pkg/hub"
)

func main() {
	st, err := hub.OpenSafeTensors("/Users/theapemachine/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6/model.safetensors")
	if err != nil {
		panic(err)
	}
	defer st.Close()
	for name := range st.Index {
		fmt.Println(name)
	}
}
