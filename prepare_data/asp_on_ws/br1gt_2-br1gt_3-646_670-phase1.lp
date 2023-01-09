
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

	orientation(br1gt_2,12,646). coords(br1gt_2,86,209,646).  visible(br1gt_2,646). happensAt(inactive(br1gt_2),646). orientation(br1gt_3,110,646). coords(br1gt_3,299,92,646).  visible(br1gt_3,646). happensAt(active(br1gt_3),646).
orientation(br1gt_2,12,647). coords(br1gt_2,86,209,647).  visible(br1gt_2,647). happensAt(inactive(br1gt_2),647). orientation(br1gt_3,110,647). coords(br1gt_3,299,92,647).  visible(br1gt_3,647). happensAt(active(br1gt_3),647).
orientation(br1gt_2,12,648). coords(br1gt_2,86,209,648).  visible(br1gt_2,648). happensAt(inactive(br1gt_2),648). orientation(br1gt_3,110,648). coords(br1gt_3,299,92,648).  visible(br1gt_3,648). happensAt(active(br1gt_3),648).
orientation(br1gt_2,12,649). coords(br1gt_2,86,209,649).  visible(br1gt_2,649). happensAt(inactive(br1gt_2),649). orientation(br1gt_3,110,649). coords(br1gt_3,299,93,649).  visible(br1gt_3,649). happensAt(active(br1gt_3),649).
orientation(br1gt_2,12,650). coords(br1gt_2,86,209,650).  visible(br1gt_2,650). happensAt(inactive(br1gt_2),650). orientation(br1gt_3,110,650). coords(br1gt_3,299,93,650).  visible(br1gt_3,650). happensAt(active(br1gt_3),650).
orientation(br1gt_2,12,651). coords(br1gt_2,86,209,651).  visible(br1gt_2,651). happensAt(inactive(br1gt_2),651). orientation(br1gt_3,110,651). coords(br1gt_3,299,93,651).  visible(br1gt_3,651). happensAt(active(br1gt_3),651).
orientation(br1gt_2,12,652). coords(br1gt_2,86,209,652).  visible(br1gt_2,652). happensAt(inactive(br1gt_2),652). orientation(br1gt_3,110,652). coords(br1gt_3,300,92,652).  visible(br1gt_3,652). happensAt(active(br1gt_3),652).
orientation(br1gt_2,12,653). coords(br1gt_2,86,209,653).  visible(br1gt_2,653). happensAt(inactive(br1gt_2),653). orientation(br1gt_3,110,653). coords(br1gt_3,300,92,653).  visible(br1gt_3,653). happensAt(active(br1gt_3),653).
orientation(br1gt_2,12,654). coords(br1gt_2,86,209,654).  visible(br1gt_2,654). happensAt(inactive(br1gt_2),654). orientation(br1gt_3,110,654). coords(br1gt_3,299,92,654).  visible(br1gt_3,654). happensAt(active(br1gt_3),654).
orientation(br1gt_2,12,655). coords(br1gt_2,86,209,655).  visible(br1gt_2,655). happensAt(inactive(br1gt_2),655). orientation(br1gt_3,110,655). coords(br1gt_3,299,92,655).  visible(br1gt_3,655). happensAt(active(br1gt_3),655).
orientation(br1gt_2,12,656). coords(br1gt_2,86,209,656).  visible(br1gt_2,656). happensAt(inactive(br1gt_2),656). orientation(br1gt_3,119,656). coords(br1gt_3,299,92,656).  visible(br1gt_3,656). happensAt(active(br1gt_3),656).
orientation(br1gt_2,12,657). coords(br1gt_2,86,209,657).  visible(br1gt_2,657). happensAt(inactive(br1gt_2),657). orientation(br1gt_3,119,657). coords(br1gt_3,299,91,657).  visible(br1gt_3,657). happensAt(active(br1gt_3),657).
orientation(br1gt_2,12,658). coords(br1gt_2,86,209,658).  visible(br1gt_2,658). happensAt(inactive(br1gt_2),658). orientation(br1gt_3,119,658). coords(br1gt_3,298,91,658).  visible(br1gt_3,658). happensAt(active(br1gt_3),658).
orientation(br1gt_2,12,659). coords(br1gt_2,86,209,659).  visible(br1gt_2,659). happensAt(inactive(br1gt_2),659). orientation(br1gt_3,119,659). coords(br1gt_3,298,91,659).  visible(br1gt_3,659). happensAt(active(br1gt_3),659).
orientation(br1gt_2,12,660). coords(br1gt_2,86,209,660).  visible(br1gt_2,660). happensAt(inactive(br1gt_2),660). orientation(br1gt_3,119,660). coords(br1gt_3,298,91,660).  visible(br1gt_3,660). happensAt(active(br1gt_3),660).
orientation(br1gt_2,12,661). coords(br1gt_2,86,209,661).  visible(br1gt_2,661). happensAt(inactive(br1gt_2),661). orientation(br1gt_3,119,661). coords(br1gt_3,298,91,661).  visible(br1gt_3,661). happensAt(active(br1gt_3),661).
orientation(br1gt_2,12,662). coords(br1gt_2,86,209,662).  visible(br1gt_2,662). happensAt(inactive(br1gt_2),662). orientation(br1gt_3,119,662). coords(br1gt_3,298,91,662).  visible(br1gt_3,662). happensAt(active(br1gt_3),662).
orientation(br1gt_2,12,663). coords(br1gt_2,86,209,663).  visible(br1gt_2,663). happensAt(inactive(br1gt_2),663). orientation(br1gt_3,119,663). coords(br1gt_3,298,92,663).  visible(br1gt_3,663). happensAt(active(br1gt_3),663).
orientation(br1gt_2,12,664). coords(br1gt_2,86,209,664).  visible(br1gt_2,664). happensAt(inactive(br1gt_2),664). orientation(br1gt_3,119,664). coords(br1gt_3,298,92,664).  visible(br1gt_3,664). happensAt(active(br1gt_3),664).
orientation(br1gt_2,12,665). coords(br1gt_2,86,209,665).  visible(br1gt_2,665). happensAt(inactive(br1gt_2),665). orientation(br1gt_3,114,665). coords(br1gt_3,300,92,665).  visible(br1gt_3,665). happensAt(active(br1gt_3),665).
orientation(br1gt_2,12,666). coords(br1gt_2,86,209,666).  visible(br1gt_2,666). happensAt(inactive(br1gt_2),666). orientation(br1gt_3,114,666). coords(br1gt_3,301,92,666).  visible(br1gt_3,666). happensAt(active(br1gt_3),666).
orientation(br1gt_2,12,667). coords(br1gt_2,86,209,667).  visible(br1gt_2,667). happensAt(inactive(br1gt_2),667). orientation(br1gt_3,114,667). coords(br1gt_3,301,92,667).  visible(br1gt_3,667). happensAt(active(br1gt_3),667).
orientation(br1gt_2,12,668). coords(br1gt_2,86,209,668).  visible(br1gt_2,668). happensAt(inactive(br1gt_2),668). orientation(br1gt_3,114,668). coords(br1gt_3,301,92,668).  visible(br1gt_3,668). happensAt(active(br1gt_3),668).
orientation(br1gt_2,12,669). coords(br1gt_2,86,209,669).  visible(br1gt_2,669). happensAt(inactive(br1gt_2),669). orientation(br1gt_3,114,669). coords(br1gt_3,301,92,669).  visible(br1gt_3,669). happensAt(active(br1gt_3),669).
orientation(br1gt_2,12,670). coords(br1gt_2,86,209,670).  visible(br1gt_2,670). happensAt(inactive(br1gt_2),670). orientation(br1gt_3,114,670). coords(br1gt_3,302,93,670).  visible(br1gt_3,670). happensAt(active(br1gt_3),670).
#show holdsAt/2.