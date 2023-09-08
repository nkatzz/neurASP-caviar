
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

	orientation(lbpugt_2,132,647). coords(lbpugt_2,184,57,647).  visible(lbpugt_2,647). happensAt(walking(lbpugt_2),647). orientation(lbpugt_4,135,647). coords(lbpugt_4,219,34,647).  visible(lbpugt_4,647). happensAt(inactive(lbpugt_4),647).
orientation(lbpugt_2,132,648). coords(lbpugt_2,185,58,648).  visible(lbpugt_2,648). happensAt(walking(lbpugt_2),648). orientation(lbpugt_4,135,648). coords(lbpugt_4,219,34,648).  visible(lbpugt_4,648). happensAt(inactive(lbpugt_4),648).
orientation(lbpugt_2,132,649). coords(lbpugt_2,185,58,649).  visible(lbpugt_2,649). happensAt(walking(lbpugt_2),649). orientation(lbpugt_4,135,649). coords(lbpugt_4,219,34,649).  visible(lbpugt_4,649). happensAt(inactive(lbpugt_4),649).
orientation(lbpugt_2,132,650). coords(lbpugt_2,185,58,650).  visible(lbpugt_2,650). happensAt(walking(lbpugt_2),650). orientation(lbpugt_4,135,650). coords(lbpugt_4,219,34,650).  visible(lbpugt_4,650). happensAt(inactive(lbpugt_4),650).
orientation(lbpugt_2,132,651). coords(lbpugt_2,186,58,651).  visible(lbpugt_2,651). happensAt(walking(lbpugt_2),651). orientation(lbpugt_4,135,651). coords(lbpugt_4,219,34,651).  visible(lbpugt_4,651). happensAt(inactive(lbpugt_4),651).
orientation(lbpugt_2,132,652). coords(lbpugt_2,188,58,652).  visible(lbpugt_2,652). happensAt(walking(lbpugt_2),652). orientation(lbpugt_4,135,652). coords(lbpugt_4,219,34,652).  visible(lbpugt_4,652). happensAt(inactive(lbpugt_4),652).
orientation(lbpugt_2,132,653). coords(lbpugt_2,190,58,653).  visible(lbpugt_2,653). happensAt(walking(lbpugt_2),653). orientation(lbpugt_4,135,653). coords(lbpugt_4,219,34,653).  visible(lbpugt_4,653). happensAt(inactive(lbpugt_4),653).
orientation(lbpugt_2,132,654). coords(lbpugt_2,192,59,654).  visible(lbpugt_2,654). happensAt(walking(lbpugt_2),654). orientation(lbpugt_4,135,654). coords(lbpugt_4,219,34,654).  visible(lbpugt_4,654). happensAt(inactive(lbpugt_4),654).
orientation(lbpugt_2,132,655). coords(lbpugt_2,193,60,655).  visible(lbpugt_2,655). happensAt(walking(lbpugt_2),655). orientation(lbpugt_4,135,655). coords(lbpugt_4,219,34,655).  visible(lbpugt_4,655). happensAt(inactive(lbpugt_4),655).
orientation(lbpugt_2,132,656). coords(lbpugt_2,194,61,656).  visible(lbpugt_2,656). happensAt(walking(lbpugt_2),656). orientation(lbpugt_4,135,656). coords(lbpugt_4,219,34,656).  visible(lbpugt_4,656). happensAt(inactive(lbpugt_4),656).
orientation(lbpugt_2,132,657). coords(lbpugt_2,194,61,657).  visible(lbpugt_2,657). happensAt(walking(lbpugt_2),657). orientation(lbpugt_4,135,657). coords(lbpugt_4,219,34,657).  visible(lbpugt_4,657). happensAt(inactive(lbpugt_4),657).
orientation(lbpugt_2,132,658). coords(lbpugt_2,196,61,658).  visible(lbpugt_2,658). happensAt(walking(lbpugt_2),658). orientation(lbpugt_4,135,658). coords(lbpugt_4,219,34,658).  visible(lbpugt_4,658). happensAt(inactive(lbpugt_4),658).
orientation(lbpugt_2,132,659). coords(lbpugt_2,197,63,659).  visible(lbpugt_2,659). happensAt(walking(lbpugt_2),659). orientation(lbpugt_4,135,659). coords(lbpugt_4,219,34,659).  visible(lbpugt_4,659). happensAt(inactive(lbpugt_4),659).
orientation(lbpugt_2,132,660). coords(lbpugt_2,198,64,660).  visible(lbpugt_2,660). happensAt(walking(lbpugt_2),660). orientation(lbpugt_4,135,660). coords(lbpugt_4,219,34,660).  visible(lbpugt_4,660). happensAt(inactive(lbpugt_4),660).
orientation(lbpugt_2,132,661). coords(lbpugt_2,198,64,661).  visible(lbpugt_2,661). happensAt(walking(lbpugt_2),661). orientation(lbpugt_4,135,661). coords(lbpugt_4,219,34,661).  visible(lbpugt_4,661). happensAt(inactive(lbpugt_4),661).
orientation(lbpugt_2,132,662). coords(lbpugt_2,199,64,662).  visible(lbpugt_2,662). happensAt(walking(lbpugt_2),662). orientation(lbpugt_4,135,662). coords(lbpugt_4,219,34,662).  visible(lbpugt_4,662). happensAt(inactive(lbpugt_4),662).
orientation(lbpugt_2,132,663). coords(lbpugt_2,199,64,663).  visible(lbpugt_2,663). happensAt(walking(lbpugt_2),663). orientation(lbpugt_4,135,663). coords(lbpugt_4,219,34,663).  visible(lbpugt_4,663). happensAt(inactive(lbpugt_4),663).
orientation(lbpugt_2,132,664). coords(lbpugt_2,199,64,664).  visible(lbpugt_2,664). happensAt(walking(lbpugt_2),664). orientation(lbpugt_4,135,664). coords(lbpugt_4,219,34,664).  visible(lbpugt_4,664). happensAt(inactive(lbpugt_4),664).
orientation(lbpugt_2,132,665). coords(lbpugt_2,200,64,665).  visible(lbpugt_2,665). happensAt(walking(lbpugt_2),665). orientation(lbpugt_4,135,665). coords(lbpugt_4,219,34,665).  visible(lbpugt_4,665). happensAt(inactive(lbpugt_4),665).
orientation(lbpugt_2,127,666). coords(lbpugt_2,200,64,666).  visible(lbpugt_2,666). happensAt(walking(lbpugt_2),666). orientation(lbpugt_4,135,666). coords(lbpugt_4,219,34,666).  visible(lbpugt_4,666). happensAt(inactive(lbpugt_4),666).
orientation(lbpugt_2,127,667). coords(lbpugt_2,200,65,667).  visible(lbpugt_2,667). happensAt(walking(lbpugt_2),667). orientation(lbpugt_4,135,667). coords(lbpugt_4,219,34,667).  visible(lbpugt_4,667). happensAt(inactive(lbpugt_4),667).
orientation(lbpugt_2,127,668). coords(lbpugt_2,202,66,668).  visible(lbpugt_2,668). happensAt(walking(lbpugt_2),668). orientation(lbpugt_4,135,668). coords(lbpugt_4,219,34,668).  visible(lbpugt_4,668). happensAt(inactive(lbpugt_4),668).
orientation(lbpugt_2,127,669). coords(lbpugt_2,204,67,669).  visible(lbpugt_2,669). happensAt(walking(lbpugt_2),669). orientation(lbpugt_4,135,669). coords(lbpugt_4,219,34,669).  visible(lbpugt_4,669). happensAt(inactive(lbpugt_4),669).
orientation(lbpugt_2,131,670). coords(lbpugt_2,207,68,670).  visible(lbpugt_2,670). happensAt(walking(lbpugt_2),670). orientation(lbpugt_4,135,670). coords(lbpugt_4,219,34,670).  visible(lbpugt_4,670). happensAt(inactive(lbpugt_4),670).
orientation(lbpugt_2,131,671). coords(lbpugt_2,210,70,671).  visible(lbpugt_2,671). happensAt(walking(lbpugt_2),671). orientation(lbpugt_4,135,671). coords(lbpugt_4,219,34,671).  visible(lbpugt_4,671). happensAt(inactive(lbpugt_4),671).
#show holdsAt/2.