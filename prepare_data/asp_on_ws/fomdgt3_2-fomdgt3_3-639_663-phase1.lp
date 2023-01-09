
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

	orientation(fomdgt3_2,130,639). coords(fomdgt3_2,272,180,639).  visible(fomdgt3_2,639). happensAt(walking(fomdgt3_2),639). orientation(fomdgt3_3,159,639). coords(fomdgt3_3,80,95,639).  visible(fomdgt3_3,639). happensAt(walking(fomdgt3_3),639).
orientation(fomdgt3_2,130,640). coords(fomdgt3_2,273,181,640).  visible(fomdgt3_2,640). happensAt(walking(fomdgt3_2),640). orientation(fomdgt3_3,159,640). coords(fomdgt3_3,80,95,640).  visible(fomdgt3_3,640). happensAt(walking(fomdgt3_3),640).
orientation(fomdgt3_2,130,641). coords(fomdgt3_2,273,182,641).  visible(fomdgt3_2,641). happensAt(walking(fomdgt3_2),641). orientation(fomdgt3_3,159,641). coords(fomdgt3_3,81,95,641).  visible(fomdgt3_3,641). happensAt(walking(fomdgt3_3),641).
orientation(fomdgt3_2,130,642). coords(fomdgt3_2,273,183,642).  visible(fomdgt3_2,642). happensAt(walking(fomdgt3_2),642). orientation(fomdgt3_3,159,642). coords(fomdgt3_3,81,95,642).  visible(fomdgt3_3,642). happensAt(walking(fomdgt3_3),642).
orientation(fomdgt3_2,130,643). coords(fomdgt3_2,274,184,643).  visible(fomdgt3_2,643). happensAt(walking(fomdgt3_2),643). orientation(fomdgt3_3,159,643). coords(fomdgt3_3,83,95,643).  visible(fomdgt3_3,643). happensAt(walking(fomdgt3_3),643).
orientation(fomdgt3_2,130,644). coords(fomdgt3_2,274,184,644).  visible(fomdgt3_2,644). happensAt(walking(fomdgt3_2),644). orientation(fomdgt3_3,159,644). coords(fomdgt3_3,83,95,644).  visible(fomdgt3_3,644). happensAt(walking(fomdgt3_3),644).
orientation(fomdgt3_2,130,645). coords(fomdgt3_2,274,184,645).  visible(fomdgt3_2,645). happensAt(walking(fomdgt3_2),645). orientation(fomdgt3_3,159,645). coords(fomdgt3_3,83,95,645).  visible(fomdgt3_3,645). happensAt(walking(fomdgt3_3),645).
orientation(fomdgt3_2,130,646). coords(fomdgt3_2,274,187,646).  visible(fomdgt3_2,646). happensAt(walking(fomdgt3_2),646). orientation(fomdgt3_3,159,646). coords(fomdgt3_3,83,95,646).  visible(fomdgt3_3,646). happensAt(walking(fomdgt3_3),646).
orientation(fomdgt3_2,130,647). coords(fomdgt3_2,274,188,647).  visible(fomdgt3_2,647). happensAt(walking(fomdgt3_2),647). orientation(fomdgt3_3,159,647). coords(fomdgt3_3,83,95,647).  visible(fomdgt3_3,647). happensAt(walking(fomdgt3_3),647).
orientation(fomdgt3_2,130,648). coords(fomdgt3_2,274,190,648).  visible(fomdgt3_2,648). happensAt(walking(fomdgt3_2),648). orientation(fomdgt3_3,159,648). coords(fomdgt3_3,83,95,648).  visible(fomdgt3_3,648). happensAt(walking(fomdgt3_3),648).
orientation(fomdgt3_2,130,649). coords(fomdgt3_2,274,191,649).  visible(fomdgt3_2,649). happensAt(walking(fomdgt3_2),649). orientation(fomdgt3_3,159,649). coords(fomdgt3_3,84,96,649).  visible(fomdgt3_3,649). happensAt(walking(fomdgt3_3),649).
orientation(fomdgt3_2,130,650). coords(fomdgt3_2,277,194,650).  visible(fomdgt3_2,650). happensAt(walking(fomdgt3_2),650). orientation(fomdgt3_3,159,650). coords(fomdgt3_3,85,96,650).  visible(fomdgt3_3,650). happensAt(walking(fomdgt3_3),650).
orientation(fomdgt3_2,130,651). coords(fomdgt3_2,280,197,651).  visible(fomdgt3_2,651). happensAt(walking(fomdgt3_2),651). orientation(fomdgt3_3,159,651). coords(fomdgt3_3,86,97,651).  visible(fomdgt3_3,651). happensAt(walking(fomdgt3_3),651).
orientation(fomdgt3_2,130,652). coords(fomdgt3_2,281,199,652).  visible(fomdgt3_2,652). happensAt(walking(fomdgt3_2),652). orientation(fomdgt3_3,159,652). coords(fomdgt3_3,87,97,652).  visible(fomdgt3_3,652). happensAt(walking(fomdgt3_3),652).
orientation(fomdgt3_2,130,653). coords(fomdgt3_2,281,203,653).  visible(fomdgt3_2,653). happensAt(walking(fomdgt3_2),653). orientation(fomdgt3_3,159,653). coords(fomdgt3_3,88,97,653).  visible(fomdgt3_3,653). happensAt(walking(fomdgt3_3),653).
orientation(fomdgt3_2,130,654). coords(fomdgt3_2,283,204,654).  visible(fomdgt3_2,654). happensAt(walking(fomdgt3_2),654). orientation(fomdgt3_3,159,654). coords(fomdgt3_3,89,97,654).  visible(fomdgt3_3,654). happensAt(walking(fomdgt3_3),654).
orientation(fomdgt3_2,130,655). coords(fomdgt3_2,283,205,655).  visible(fomdgt3_2,655). happensAt(walking(fomdgt3_2),655). orientation(fomdgt3_3,159,655). coords(fomdgt3_3,89,97,655).  visible(fomdgt3_3,655). happensAt(walking(fomdgt3_3),655).
orientation(fomdgt3_2,130,656). coords(fomdgt3_2,284,205,656).  visible(fomdgt3_2,656). happensAt(walking(fomdgt3_2),656). orientation(fomdgt3_3,159,656). coords(fomdgt3_3,89,97,656).  visible(fomdgt3_3,656). happensAt(walking(fomdgt3_3),656).
orientation(fomdgt3_2,130,657). coords(fomdgt3_2,284,206,657).  visible(fomdgt3_2,657). happensAt(walking(fomdgt3_2),657). orientation(fomdgt3_3,159,657). coords(fomdgt3_3,89,97,657).  visible(fomdgt3_3,657). happensAt(walking(fomdgt3_3),657).
orientation(fomdgt3_2,130,658). coords(fomdgt3_2,284,206,658).  visible(fomdgt3_2,658). happensAt(walking(fomdgt3_2),658). orientation(fomdgt3_3,159,658). coords(fomdgt3_3,89,97,658).  visible(fomdgt3_3,658). happensAt(walking(fomdgt3_3),658).
orientation(fomdgt3_2,130,659). coords(fomdgt3_2,286,208,659).  visible(fomdgt3_2,659). happensAt(walking(fomdgt3_2),659). orientation(fomdgt3_3,159,659). coords(fomdgt3_3,89,97,659).  visible(fomdgt3_3,659). happensAt(walking(fomdgt3_3),659).
orientation(fomdgt3_2,130,660). coords(fomdgt3_2,286,208,660).  visible(fomdgt3_2,660). happensAt(walking(fomdgt3_2),660). orientation(fomdgt3_3,159,660). coords(fomdgt3_3,91,97,660).  visible(fomdgt3_3,660). happensAt(walking(fomdgt3_3),660).
orientation(fomdgt3_2,110,661). coords(fomdgt3_2,287,211,661).  visible(fomdgt3_2,661). happensAt(walking(fomdgt3_2),661). orientation(fomdgt3_3,159,661). coords(fomdgt3_3,92,97,661).  visible(fomdgt3_3,661). happensAt(walking(fomdgt3_3),661).
orientation(fomdgt3_2,110,662). coords(fomdgt3_2,288,215,662).  visible(fomdgt3_2,662). happensAt(walking(fomdgt3_2),662). orientation(fomdgt3_3,159,662). coords(fomdgt3_3,94,97,662).  visible(fomdgt3_3,662). happensAt(walking(fomdgt3_3),662).
orientation(fomdgt3_2,110,663). coords(fomdgt3_2,288,218,663).  visible(fomdgt3_2,663). happensAt(walking(fomdgt3_2),663). orientation(fomdgt3_3,159,663). coords(fomdgt3_3,95,97,663).  visible(fomdgt3_3,663). happensAt(walking(fomdgt3_3),663).
#show holdsAt/2.