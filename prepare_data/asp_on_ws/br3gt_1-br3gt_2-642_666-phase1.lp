
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

	orientation(br3gt_1,131,642). coords(br3gt_1,184,46,642).  visible(br3gt_1,642). happensAt(active(br3gt_1),642). orientation(br3gt_2,145,642). coords(br3gt_2,3,88,642).  visible(br3gt_2,642). happensAt(walking(br3gt_2),642).
orientation(br3gt_1,131,643). coords(br3gt_1,184,46,643).  visible(br3gt_1,643). happensAt(active(br3gt_1),643). orientation(br3gt_2,145,643). coords(br3gt_2,4,85,643).  visible(br3gt_2,643). happensAt(walking(br3gt_2),643).
orientation(br3gt_1,131,644). coords(br3gt_1,184,46,644).  visible(br3gt_1,644). happensAt(active(br3gt_1),644). orientation(br3gt_2,150,644). coords(br3gt_2,5,84,644).  visible(br3gt_2,644). happensAt(walking(br3gt_2),644).
orientation(br3gt_1,131,645). coords(br3gt_1,184,46,645).  visible(br3gt_1,645). happensAt(active(br3gt_1),645). orientation(br3gt_2,150,645). coords(br3gt_2,5,84,645).  visible(br3gt_2,645). happensAt(walking(br3gt_2),645).
orientation(br3gt_1,131,646). coords(br3gt_1,184,46,646).  visible(br3gt_1,646). happensAt(active(br3gt_1),646). orientation(br3gt_2,150,646). coords(br3gt_2,5,83,646).  visible(br3gt_2,646). happensAt(walking(br3gt_2),646).
orientation(br3gt_1,131,647). coords(br3gt_1,184,46,647).  visible(br3gt_1,647). happensAt(active(br3gt_1),647). orientation(br3gt_2,150,647). coords(br3gt_2,5,83,647).  visible(br3gt_2,647). happensAt(walking(br3gt_2),647).
orientation(br3gt_1,131,648). coords(br3gt_1,184,46,648).  visible(br3gt_1,648). happensAt(active(br3gt_1),648). orientation(br3gt_2,150,648). coords(br3gt_2,5,81,648).  visible(br3gt_2,648). happensAt(walking(br3gt_2),648).
orientation(br3gt_1,131,649). coords(br3gt_1,184,46,649).  visible(br3gt_1,649). happensAt(active(br3gt_1),649). orientation(br3gt_2,150,649). coords(br3gt_2,5,81,649).  visible(br3gt_2,649). happensAt(walking(br3gt_2),649).
orientation(br3gt_1,131,650). coords(br3gt_1,184,46,650).  visible(br3gt_1,650). happensAt(active(br3gt_1),650). orientation(br3gt_2,150,650). coords(br3gt_2,6,80,650).  visible(br3gt_2,650). happensAt(walking(br3gt_2),650).
orientation(br3gt_1,131,651). coords(br3gt_1,184,46,651).  visible(br3gt_1,651). happensAt(active(br3gt_1),651). orientation(br3gt_2,150,651). coords(br3gt_2,6,79,651).  visible(br3gt_2,651). happensAt(walking(br3gt_2),651).
orientation(br3gt_1,131,652). coords(br3gt_1,184,46,652).  visible(br3gt_1,652). happensAt(active(br3gt_1),652). orientation(br3gt_2,150,652). coords(br3gt_2,6,77,652).  visible(br3gt_2,652). happensAt(walking(br3gt_2),652).
orientation(br3gt_1,131,653). coords(br3gt_1,184,46,653).  visible(br3gt_1,653). happensAt(active(br3gt_1),653). orientation(br3gt_2,150,653). coords(br3gt_2,6,77,653).  visible(br3gt_2,653). happensAt(walking(br3gt_2),653).
orientation(br3gt_1,131,654). coords(br3gt_1,184,47,654).  visible(br3gt_1,654). happensAt(active(br3gt_1),654). orientation(br3gt_2,150,654). coords(br3gt_2,6,77,654).  visible(br3gt_2,654). happensAt(walking(br3gt_2),654).
orientation(br3gt_1,131,655). coords(br3gt_1,184,47,655).  visible(br3gt_1,655). happensAt(active(br3gt_1),655). orientation(br3gt_2,148,655). coords(br3gt_2,6,77,655).  visible(br3gt_2,655). happensAt(walking(br3gt_2),655).
orientation(br3gt_1,131,656). coords(br3gt_1,184,47,656).  visible(br3gt_1,656). happensAt(active(br3gt_1),656). orientation(br3gt_2,148,656). coords(br3gt_2,6,77,656).  visible(br3gt_2,656). happensAt(walking(br3gt_2),656).
orientation(br3gt_1,131,657). coords(br3gt_1,184,47,657).  visible(br3gt_1,657). happensAt(active(br3gt_1),657). orientation(br3gt_2,148,657). coords(br3gt_2,6,77,657).  visible(br3gt_2,657). happensAt(walking(br3gt_2),657).
orientation(br3gt_1,131,658). coords(br3gt_1,184,47,658).  visible(br3gt_1,658). happensAt(active(br3gt_1),658). orientation(br3gt_2,148,658). coords(br3gt_2,7,74,658).  visible(br3gt_2,658). happensAt(walking(br3gt_2),658).
orientation(br3gt_1,131,659). coords(br3gt_1,184,47,659).  visible(br3gt_1,659). happensAt(active(br3gt_1),659). orientation(br3gt_2,148,659). coords(br3gt_2,7,73,659).  visible(br3gt_2,659). happensAt(walking(br3gt_2),659).
orientation(br3gt_1,131,660). coords(br3gt_1,184,47,660).  visible(br3gt_1,660). happensAt(walking(br3gt_1),660). orientation(br3gt_2,148,660). coords(br3gt_2,7,73,660).  visible(br3gt_2,660). happensAt(walking(br3gt_2),660).
orientation(br3gt_1,131,661). coords(br3gt_1,184,47,661).  visible(br3gt_1,661). happensAt(walking(br3gt_1),661). orientation(br3gt_2,148,661). coords(br3gt_2,7,72,661).  visible(br3gt_2,661). happensAt(walking(br3gt_2),661).
orientation(br3gt_1,131,662). coords(br3gt_1,184,46,662).  visible(br3gt_1,662). happensAt(walking(br3gt_1),662). orientation(br3gt_2,148,662). coords(br3gt_2,7,71,662).  visible(br3gt_2,662). happensAt(walking(br3gt_2),662).
orientation(br3gt_1,131,663). coords(br3gt_1,184,46,663).  visible(br3gt_1,663). happensAt(walking(br3gt_1),663). orientation(br3gt_2,148,663). coords(br3gt_2,7,69,663).  visible(br3gt_2,663). happensAt(walking(br3gt_2),663).
orientation(br3gt_1,131,664). coords(br3gt_1,185,46,664).  visible(br3gt_1,664). happensAt(walking(br3gt_1),664). orientation(br3gt_2,148,664). coords(br3gt_2,7,69,664).  visible(br3gt_2,664). happensAt(walking(br3gt_2),664).
orientation(br3gt_1,131,665). coords(br3gt_1,185,46,665).  visible(br3gt_1,665). happensAt(walking(br3gt_1),665). orientation(br3gt_2,148,665). coords(br3gt_2,8,68,665).  visible(br3gt_2,665). happensAt(walking(br3gt_2),665).
orientation(br3gt_1,131,666). coords(br3gt_1,185,46,666).  visible(br3gt_1,666). happensAt(walking(br3gt_1),666). orientation(br3gt_2,148,666). coords(br3gt_2,8,68,666).  visible(br3gt_2,666). happensAt(walking(br3gt_2),666).
#show holdsAt/2.