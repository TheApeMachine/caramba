package cmd

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	
	"github.com/spf13/cobra"
	"github.com/theapemachine/caramba/pkg/agent/state"
	"github.com/theapemachine/errnie"
)

var stateManager state.Manager

/*
stateCmd is a command that manages agent states.
*/
var stateCmd = &cobra.Command{
	Use:   "state",
	Short: "Manage agent states",
	Long:  `Commands for managing agent states including creation, retrieval, and deletion`,
	PersistentPreRun: func(cmd *cobra.Command, args []string) {
		// Initialize logger
		os.Setenv("LOG_LEVEL", "debug")
		os.Setenv("LOGFILE", "true")
		errnie.InitLogger()
		
		// Initialize state manager if not already done
		if stateManager == nil {
			stateManager = state.NewInMemoryStateManager()
		}
	},
}

/*
createStateCmd is a command that creates a new agent state.
*/
var createStateCmd = &cobra.Command{
	Use:   "create",
	Short: "Create a new agent state",
	Long:  `Creates a new agent state with the specified name`,
	Run: func(cmd *cobra.Command, args []string) {
		name, _ := cmd.Flags().GetString("name")
		
		id, err := stateManager.CreateState(context.Background(), name)
		if err != nil {
			errnie.Error(err)
			return
		}
		
		fmt.Printf("Created state with ID: %s\n", id)
	},
}

/*
getStateCmd is a command that retrieves an agent state.
*/
var getStateCmd = &cobra.Command{
	Use:   "get",
	Short: "Get an agent state",
	Long:  `Retrieves an agent state by ID`,
	Run: func(cmd *cobra.Command, args []string) {
		id, _ := cmd.Flags().GetString("id")
		
		state, err := stateManager.GetState(context.Background(), id)
		if err != nil {
			errnie.Error(err)
			return
		}
		
		// Print the state as JSON
		jsonData, err := json.MarshalIndent(state, "", "  ")
		if err != nil {
			errnie.Error(err)
			return
		}
		
		fmt.Println(string(jsonData))
	},
}

/*
updateStateCmd is a command that updates an agent state.
*/
var updateStateCmd = &cobra.Command{
	Use:   "update",
	Short: "Update an agent state",
	Long:  `Updates an agent state with new values`,
	Run: func(cmd *cobra.Command, args []string) {
		id, _ := cmd.Flags().GetString("id")
		status, _ := cmd.Flags().GetString("status")
		task, _ := cmd.Flags().GetString("task")
		
		err := stateManager.UpdateState(context.Background(), id, func(s *state.State) error {
			if status != "" {
				s.Status = status
			}
			
			if task != "" {
				s.CurrentTask = task
			}
			
			// Add a history entry
			s.History = append(s.History, state.HistoryEntry{
				Timestamp: state.Now(),
				Action:    "update",
				Details: map[string]interface{}{
					"status": status,
					"task":   task,
				},
			})
			
			return nil
		})
		
		if err != nil {
			errnie.Error(err)
			return
		}
		
		fmt.Printf("Updated state with ID: %s\n", id)
	},
}

/*
deleteStateCmd is a command that deletes an agent state.
*/
var deleteStateCmd = &cobra.Command{
	Use:   "delete",
	Short: "Delete an agent state",
	Long:  `Deletes an agent state by ID`,
	Run: func(cmd *cobra.Command, args []string) {
		id, _ := cmd.Flags().GetString("id")
		
		err := stateManager.DeleteState(context.Background(), id)
		if err != nil {
			errnie.Error(err)
			return
		}
		
		fmt.Printf("Deleted state with ID: %s\n", id)
	},
}

/*
listStatesCmd is a command that lists all agent states.
*/
var listStatesCmd = &cobra.Command{
	Use:   "list",
	Short: "List all agent states",
	Long:  `Lists all agent states`,
	Run: func(cmd *cobra.Command, args []string) {
		states, err := stateManager.ListStates(context.Background())
		if err != nil {
			errnie.Error(err)
			return
		}
		
		fmt.Printf("Found %d states:\n\n", len(states))
		
		for _, s := range states {
			fmt.Printf("ID: %s\n", s.ID)
			fmt.Printf("Name: %s\n", s.Name)
			fmt.Printf("Status: %s\n", s.Status)
			fmt.Printf("Created: %s\n", s.Created.Format("2006-01-02 15:04:05"))
			fmt.Printf("Last Modified: %s\n\n", s.LastModified.Format("2006-01-02 15:04:05"))
		}
	},
}

func init() {
	rootCmd.AddCommand(stateCmd)
	
	// Add subcommands
	stateCmd.AddCommand(createStateCmd)
	stateCmd.AddCommand(getStateCmd)
	stateCmd.AddCommand(updateStateCmd)
	stateCmd.AddCommand(deleteStateCmd)
	stateCmd.AddCommand(listStatesCmd)
	
	// Add flags for create command
	createStateCmd.Flags().StringP("name", "n", "Agent", "Name of the agent state")
	
	// Add flags for get command
	getStateCmd.Flags().StringP("id", "i", "", "ID of the agent state")
	getStateCmd.MarkFlagRequired("id")
	
	// Add flags for update command
	updateStateCmd.Flags().StringP("id", "i", "", "ID of the agent state")
	updateStateCmd.Flags().StringP("status", "s", "", "New status for the agent")
	updateStateCmd.Flags().StringP("task", "t", "", "New current task for the agent")
	updateStateCmd.MarkFlagRequired("id")
	
	// Add flags for delete command
	deleteStateCmd.Flags().StringP("id", "i", "", "ID of the agent state")
	deleteStateCmd.MarkFlagRequired("id")
}
