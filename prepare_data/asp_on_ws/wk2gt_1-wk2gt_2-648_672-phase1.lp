
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

	orientation(wk2gt_1,125,648). coords(wk2gt_1,114,53,648).  visible(wk2gt_1,648). happensAt(inactive(wk2gt_1),648). orientation(wk2gt_2,142,648). coords(wk2gt_2,99,63,648).  visible(wk2gt_2,648). happensAt(inactive(wk2gt_2),648).
orientation(wk2gt_1,125,649). coords(wk2gt_1,114,53,649).  visible(wk2gt_1,649). happensAt(inactive(wk2gt_1),649). orientation(wk2gt_2,142,649). coords(wk2gt_2,99,63,649).  visible(wk2gt_2,649). happensAt(inactive(wk2gt_2),649).
orientation(wk2gt_1,125,650). coords(wk2gt_1,114,53,650).  visible(wk2gt_1,650). happensAt(inactive(wk2gt_1),650). orientation(wk2gt_2,142,650). coords(wk2gt_2,99,63,650).  visible(wk2gt_2,650). happensAt(inactive(wk2gt_2),650).
orientation(wk2gt_1,125,651). coords(wk2gt_1,114,53,651).  visible(wk2gt_1,651). happensAt(inactive(wk2gt_1),651). orientation(wk2gt_2,142,651). coords(wk2gt_2,99,63,651).  visible(wk2gt_2,651). happensAt(inactive(wk2gt_2),651).
orientation(wk2gt_1,125,652). coords(wk2gt_1,114,53,652).  visible(wk2gt_1,652). happensAt(inactive(wk2gt_1),652). orientation(wk2gt_2,142,652). coords(wk2gt_2,99,63,652).  visible(wk2gt_2,652). happensAt(inactive(wk2gt_2),652).
orientation(wk2gt_1,125,653). coords(wk2gt_1,114,53,653).  visible(wk2gt_1,653). happensAt(inactive(wk2gt_1),653). orientation(wk2gt_2,142,653). coords(wk2gt_2,99,63,653).  visible(wk2gt_2,653). happensAt(inactive(wk2gt_2),653).
orientation(wk2gt_1,125,654). coords(wk2gt_1,114,53,654).  visible(wk2gt_1,654). happensAt(inactive(wk2gt_1),654). orientation(wk2gt_2,142,654). coords(wk2gt_2,99,63,654).  visible(wk2gt_2,654). happensAt(inactive(wk2gt_2),654).
orientation(wk2gt_1,125,655). coords(wk2gt_1,114,53,655).  visible(wk2gt_1,655). happensAt(inactive(wk2gt_1),655). orientation(wk2gt_2,142,655). coords(wk2gt_2,99,63,655).  visible(wk2gt_2,655). happensAt(inactive(wk2gt_2),655).
orientation(wk2gt_1,125,656). coords(wk2gt_1,114,53,656).  visible(wk2gt_1,656). happensAt(inactive(wk2gt_1),656). orientation(wk2gt_2,142,656). coords(wk2gt_2,99,63,656).  visible(wk2gt_2,656). happensAt(inactive(wk2gt_2),656).
orientation(wk2gt_1,125,657). coords(wk2gt_1,114,53,657).  visible(wk2gt_1,657). happensAt(inactive(wk2gt_1),657). orientation(wk2gt_2,142,657). coords(wk2gt_2,99,63,657).  visible(wk2gt_2,657). happensAt(inactive(wk2gt_2),657).
orientation(wk2gt_1,125,658). coords(wk2gt_1,114,53,658).  visible(wk2gt_1,658). happensAt(inactive(wk2gt_1),658). orientation(wk2gt_2,142,658). coords(wk2gt_2,99,63,658).  visible(wk2gt_2,658). happensAt(inactive(wk2gt_2),658).
orientation(wk2gt_1,125,659). coords(wk2gt_1,114,53,659).  visible(wk2gt_1,659). happensAt(inactive(wk2gt_1),659). orientation(wk2gt_2,142,659). coords(wk2gt_2,99,63,659).  visible(wk2gt_2,659). happensAt(inactive(wk2gt_2),659).
orientation(wk2gt_1,125,660). coords(wk2gt_1,114,53,660).  visible(wk2gt_1,660). happensAt(inactive(wk2gt_1),660). orientation(wk2gt_2,142,660). coords(wk2gt_2,99,63,660).  visible(wk2gt_2,660). happensAt(inactive(wk2gt_2),660).
orientation(wk2gt_1,125,661). coords(wk2gt_1,114,53,661).  visible(wk2gt_1,661). happensAt(inactive(wk2gt_1),661). orientation(wk2gt_2,142,661). coords(wk2gt_2,99,63,661).  visible(wk2gt_2,661). happensAt(inactive(wk2gt_2),661).
orientation(wk2gt_1,125,662). coords(wk2gt_1,114,53,662).  visible(wk2gt_1,662). happensAt(inactive(wk2gt_1),662). orientation(wk2gt_2,142,662). coords(wk2gt_2,99,63,662).  visible(wk2gt_2,662). happensAt(inactive(wk2gt_2),662).
orientation(wk2gt_1,125,663). coords(wk2gt_1,114,53,663).  visible(wk2gt_1,663). happensAt(inactive(wk2gt_1),663). orientation(wk2gt_2,142,663). coords(wk2gt_2,99,63,663).  visible(wk2gt_2,663). happensAt(inactive(wk2gt_2),663).
orientation(wk2gt_1,125,664). coords(wk2gt_1,114,53,664).  visible(wk2gt_1,664). happensAt(inactive(wk2gt_1),664). orientation(wk2gt_2,142,664). coords(wk2gt_2,99,63,664).  visible(wk2gt_2,664). happensAt(inactive(wk2gt_2),664).
orientation(wk2gt_1,125,665). coords(wk2gt_1,114,53,665).  visible(wk2gt_1,665). happensAt(inactive(wk2gt_1),665). orientation(wk2gt_2,142,665). coords(wk2gt_2,99,63,665).  visible(wk2gt_2,665). happensAt(inactive(wk2gt_2),665).
orientation(wk2gt_1,125,666). coords(wk2gt_1,114,53,666).  visible(wk2gt_1,666). happensAt(inactive(wk2gt_1),666). orientation(wk2gt_2,142,666). coords(wk2gt_2,99,63,666).  visible(wk2gt_2,666). happensAt(inactive(wk2gt_2),666).
orientation(wk2gt_1,125,667). coords(wk2gt_1,114,53,667).  visible(wk2gt_1,667). happensAt(inactive(wk2gt_1),667). orientation(wk2gt_2,142,667). coords(wk2gt_2,99,63,667).  visible(wk2gt_2,667). happensAt(inactive(wk2gt_2),667).
orientation(wk2gt_1,125,668). coords(wk2gt_1,114,53,668).  visible(wk2gt_1,668). happensAt(inactive(wk2gt_1),668). orientation(wk2gt_2,142,668). coords(wk2gt_2,99,63,668).  visible(wk2gt_2,668). happensAt(inactive(wk2gt_2),668).
orientation(wk2gt_1,125,669). coords(wk2gt_1,114,53,669).  visible(wk2gt_1,669). happensAt(inactive(wk2gt_1),669). orientation(wk2gt_2,142,669). coords(wk2gt_2,99,63,669).  visible(wk2gt_2,669). happensAt(inactive(wk2gt_2),669).
orientation(wk2gt_1,125,670). coords(wk2gt_1,114,53,670).  visible(wk2gt_1,670). happensAt(inactive(wk2gt_1),670). orientation(wk2gt_2,142,670). coords(wk2gt_2,99,63,670).  visible(wk2gt_2,670). happensAt(inactive(wk2gt_2),670).
orientation(wk2gt_1,125,671). coords(wk2gt_1,114,53,671).  visible(wk2gt_1,671). happensAt(inactive(wk2gt_1),671). orientation(wk2gt_2,142,671). coords(wk2gt_2,99,63,671).  visible(wk2gt_2,671). happensAt(inactive(wk2gt_2),671).
orientation(wk2gt_1,125,672). coords(wk2gt_1,114,53,672).  visible(wk2gt_1,672). happensAt(inactive(wk2gt_1),672). orientation(wk2gt_2,142,672). coords(wk2gt_2,99,63,672).  visible(wk2gt_2,672). happensAt(inactive(wk2gt_2),672).
#show holdsAt/2.