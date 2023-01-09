
holdsAt(F,Te) :- initiatedAt(F,Ts), fluent(F), next(Ts,Te).
holdsAt(F,Te) :- holdsAt(F,Ts), not terminatedAt(F,Ts), fluent(F), next(Ts,Te).


fluent(moving(X,Y)) :- person(X), person(Y), X != Y.
fluent(meeting(X,Y)) :- person(X), person(Y), X != Y.


#include "../next.lp".

% Distance/orientation calculations:
#include "../distances.lp".

% Definitions of "type" predicates (extracted from the data):
time(T) :- happensAt(disappear(_),T).
time(T) :- happensAt(appear(_),T).
time(T) :- happensAt(active(_),T).
time(T) :- happensAt(inactive(_),T).
time(T) :- happensAt(walking(_),T).
time(T) :- happensAt(running(_),T).
time(T) :- coords(_,_,_,T).
time(T) :- orientation(_,_,T).


person(X) :- happensAt(disappear(X),_).
person(X) :- happensAt(appear(X),_).
person(X) :- happensAt(active(X),_).
person(X) :- happensAt(inactive(X),_).
person(X) :- happensAt(walking(X),_).
person(X) :- happensAt(running(X),_).
person(X) :- coords(X,_,_,_).
person(X) :- orientation(X,_,_).


% Definition for the "moving" complex event: 
initiatedAt(moving(X0,X1),X2) :- happensAt(walking(X0),X2),happensAt(walking(X1),X2),orientationMove(X0,X1,X2),close(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(active(X1),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(inactive(X1),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X1),X2),happensAt(inactive(X0),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(running(X0),X2), person(X1).
terminatedAt(moving(X0,X1),X2) :- happensAt(running(X1),X2), person(X0).
terminatedAt(moving(X0,X1),X2) :- happensAt(disappear(X0),X2), person(X1).
terminatedAt(moving(X0,X1),X2) :- happensAt(disappear(X1),X2), person(X0).

% Definition for the "meeting" complex event:
initiatedAt(meeting(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(active(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(inactive(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(inactive(X0),X2),happensAt(active(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(inactive(X0),X2),happensAt(inactive(X1),X2),close(X0,X1,25,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(running(X0),X2),person(X1).
terminatedAt(meeting(X0,X1),X2) :- happensAt(running(X1),X2),person(X0).
terminatedAt(meeting(X0,X1),X2) :- happensAt(disappear(X0),X2), person(X1).
terminatedAt(meeting(X0,X1),X2) :- happensAt(disappear(X1),X2), person(X0).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,25,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,25,X2).

	orientation(lbpugt_2,121,971). coords(lbpugt_2,190,51,971).  visible(lbpugt_2,971). happensAt(walking(lbpugt_2),971). orientation(lbpugt_4,135,971). coords(lbpugt_4,219,34,971).  visible(lbpugt_4,971). happensAt(inactive(lbpugt_4),971).
orientation(lbpugt_2,121,972). coords(lbpugt_2,191,51,972).  visible(lbpugt_2,972). happensAt(walking(lbpugt_2),972). orientation(lbpugt_4,135,972). coords(lbpugt_4,219,34,972).  visible(lbpugt_4,972). happensAt(inactive(lbpugt_4),972).
orientation(lbpugt_2,121,973). coords(lbpugt_2,191,50,973).  visible(lbpugt_2,973). happensAt(walking(lbpugt_2),973). orientation(lbpugt_4,135,973). coords(lbpugt_4,219,34,973).  visible(lbpugt_4,973). happensAt(inactive(lbpugt_4),973).
orientation(lbpugt_2,121,974). coords(lbpugt_2,192,50,974).  visible(lbpugt_2,974). happensAt(walking(lbpugt_2),974). orientation(lbpugt_4,135,974). coords(lbpugt_4,219,34,974).  visible(lbpugt_4,974). happensAt(inactive(lbpugt_4),974).
orientation(lbpugt_2,121,975). coords(lbpugt_2,192,49,975).  visible(lbpugt_2,975). happensAt(walking(lbpugt_2),975). orientation(lbpugt_4,135,975). coords(lbpugt_4,219,34,975).  visible(lbpugt_4,975). happensAt(inactive(lbpugt_4),975).
orientation(lbpugt_2,121,976). coords(lbpugt_2,192,47,976).  visible(lbpugt_2,976). happensAt(walking(lbpugt_2),976). orientation(lbpugt_4,135,976). coords(lbpugt_4,219,34,976).  visible(lbpugt_4,976). happensAt(inactive(lbpugt_4),976).
orientation(lbpugt_2,128,977). coords(lbpugt_2,192,46,977).  visible(lbpugt_2,977). happensAt(walking(lbpugt_2),977). orientation(lbpugt_4,135,977). coords(lbpugt_4,219,34,977).  visible(lbpugt_4,977). happensAt(inactive(lbpugt_4),977).
orientation(lbpugt_2,128,978). coords(lbpugt_2,192,44,978).  visible(lbpugt_2,978). happensAt(walking(lbpugt_2),978). orientation(lbpugt_4,135,978). coords(lbpugt_4,219,34,978).  visible(lbpugt_4,978). happensAt(inactive(lbpugt_4),978).
orientation(lbpugt_2,128,979). coords(lbpugt_2,192,44,979).  visible(lbpugt_2,979). happensAt(walking(lbpugt_2),979). orientation(lbpugt_4,135,979). coords(lbpugt_4,219,34,979).  visible(lbpugt_4,979). happensAt(inactive(lbpugt_4),979).
orientation(lbpugt_2,128,980). coords(lbpugt_2,194,43,980).  visible(lbpugt_2,980). happensAt(walking(lbpugt_2),980). orientation(lbpugt_4,135,980). coords(lbpugt_4,219,34,980).  visible(lbpugt_4,980). happensAt(inactive(lbpugt_4),980).
orientation(lbpugt_2,128,981). coords(lbpugt_2,196,42,981).  visible(lbpugt_2,981). happensAt(walking(lbpugt_2),981). orientation(lbpugt_4,135,981). coords(lbpugt_4,219,34,981).  visible(lbpugt_4,981). happensAt(inactive(lbpugt_4),981).
orientation(lbpugt_2,128,982). coords(lbpugt_2,196,41,982).  visible(lbpugt_2,982). happensAt(walking(lbpugt_2),982). orientation(lbpugt_4,135,982). coords(lbpugt_4,219,34,982).  visible(lbpugt_4,982). happensAt(inactive(lbpugt_4),982).
orientation(lbpugt_2,128,983). coords(lbpugt_2,197,41,983).  visible(lbpugt_2,983). happensAt(walking(lbpugt_2),983). orientation(lbpugt_4,135,983). coords(lbpugt_4,219,34,983).  visible(lbpugt_4,983). happensAt(inactive(lbpugt_4),983).
orientation(lbpugt_2,121,984). coords(lbpugt_2,198,40,984).  visible(lbpugt_2,984). happensAt(walking(lbpugt_2),984). orientation(lbpugt_4,135,984). coords(lbpugt_4,219,34,984).  visible(lbpugt_4,984). happensAt(inactive(lbpugt_4),984).
orientation(lbpugt_2,123,985). coords(lbpugt_2,198,39,985).  visible(lbpugt_2,985). happensAt(walking(lbpugt_2),985). orientation(lbpugt_4,135,985). coords(lbpugt_4,219,34,985).  visible(lbpugt_4,985). happensAt(inactive(lbpugt_4),985).
orientation(lbpugt_2,119,986). coords(lbpugt_2,199,39,986).  visible(lbpugt_2,986). happensAt(walking(lbpugt_2),986). orientation(lbpugt_4,135,986). coords(lbpugt_4,219,34,986).  visible(lbpugt_4,986). happensAt(inactive(lbpugt_4),986).
orientation(lbpugt_2,119,987). coords(lbpugt_2,199,39,987).  visible(lbpugt_2,987). happensAt(walking(lbpugt_2),987). orientation(lbpugt_4,135,987). coords(lbpugt_4,219,34,987).  visible(lbpugt_4,987). happensAt(inactive(lbpugt_4),987).
orientation(lbpugt_2,115,988). coords(lbpugt_2,200,38,988).  visible(lbpugt_2,988). happensAt(walking(lbpugt_2),988). orientation(lbpugt_4,135,988). coords(lbpugt_4,219,34,988).  visible(lbpugt_4,988). happensAt(inactive(lbpugt_4),988).
orientation(lbpugt_2,111,989). coords(lbpugt_2,202,38,989).  visible(lbpugt_2,989). happensAt(walking(lbpugt_2),989). orientation(lbpugt_4,135,989). coords(lbpugt_4,219,34,989).  visible(lbpugt_4,989). happensAt(inactive(lbpugt_4),989).
orientation(lbpugt_2,105,990). coords(lbpugt_2,202,38,990).  visible(lbpugt_2,990). happensAt(walking(lbpugt_2),990). orientation(lbpugt_4,135,990). coords(lbpugt_4,219,34,990).  visible(lbpugt_4,990). happensAt(inactive(lbpugt_4),990).
orientation(lbpugt_2,102,991). coords(lbpugt_2,202,38,991).  visible(lbpugt_2,991). happensAt(walking(lbpugt_2),991). orientation(lbpugt_4,135,991). coords(lbpugt_4,219,34,991).  visible(lbpugt_4,991). happensAt(inactive(lbpugt_4),991).
orientation(lbpugt_2,96,992). coords(lbpugt_2,203,38,992).  visible(lbpugt_2,992). happensAt(walking(lbpugt_2),992). orientation(lbpugt_4,135,992). coords(lbpugt_4,219,34,992).  visible(lbpugt_4,992). happensAt(inactive(lbpugt_4),992).
orientation(lbpugt_2,102,993). coords(lbpugt_2,203,38,993).  visible(lbpugt_2,993). happensAt(active(lbpugt_2),993). orientation(lbpugt_4,135,993). coords(lbpugt_4,219,34,993).  visible(lbpugt_4,993). happensAt(inactive(lbpugt_4),993).
orientation(lbpugt_2,92,994). coords(lbpugt_2,204,38,994).  visible(lbpugt_2,994). happensAt(active(lbpugt_2),994). orientation(lbpugt_4,135,994). coords(lbpugt_4,219,34,994).  visible(lbpugt_4,994). happensAt(inactive(lbpugt_4),994).
orientation(lbpugt_2,92,995). coords(lbpugt_2,204,38,995).  visible(lbpugt_2,995). happensAt(active(lbpugt_2),995). orientation(lbpugt_4,135,995). coords(lbpugt_4,219,34,995).  visible(lbpugt_4,995). happensAt(inactive(lbpugt_4),995).
#show holdsAt/2.