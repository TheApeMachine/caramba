import { useQuery } from "@tanstack/react-query"

export const getComputeOperations = async () => {
    const query = useQuery({ 
        queryKey: ['compute'], 
        queryFn: () => fetch('/api/compute').then(res => res.json()),
    })
    
    return query.data
}