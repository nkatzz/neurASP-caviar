
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

	orientation(mwt1gt_5,150,637). coords(mwt1gt_5,82,71,637).  visible(mwt1gt_5,637). happensAt(walking(mwt1gt_5),637). orientation(mwt1gt_6,150,637). coords(mwt1gt_6,78,73,637).  visible(mwt1gt_6,637). happensAt(walking(mwt1gt_6),637).
orientation(mwt1gt_5,150,638). coords(mwt1gt_5,82,71,638).  visible(mwt1gt_5,638). happensAt(walking(mwt1gt_5),638). orientation(mwt1gt_6,150,638). coords(mwt1gt_6,78,73,638).  visible(mwt1gt_6,638). happensAt(walking(mwt1gt_6),638).
orientation(mwt1gt_5,150,639). coords(mwt1gt_5,82,71,639).  visible(mwt1gt_5,639). happensAt(walking(mwt1gt_5),639). orientation(mwt1gt_6,150,639). coords(mwt1gt_6,78,73,639).  visible(mwt1gt_6,639). happensAt(walking(mwt1gt_6),639).
orientation(mwt1gt_5,150,640). coords(mwt1gt_5,82,71,640).  visible(mwt1gt_5,640). happensAt(walking(mwt1gt_5),640). orientation(mwt1gt_6,150,640). coords(mwt1gt_6,78,73,640).  visible(mwt1gt_6,640). happensAt(walking(mwt1gt_6),640).
orientation(mwt1gt_5,150,641). coords(mwt1gt_5,82,71,641).  visible(mwt1gt_5,641). happensAt(walking(mwt1gt_5),641). orientation(mwt1gt_6,150,641). coords(mwt1gt_6,78,73,641).  visible(mwt1gt_6,641). happensAt(walking(mwt1gt_6),641).
orientation(mwt1gt_5,150,642). coords(mwt1gt_5,82,71,642).  visible(mwt1gt_5,642). happensAt(walking(mwt1gt_5),642). orientation(mwt1gt_6,150,642). coords(mwt1gt_6,78,73,642).  visible(mwt1gt_6,642). happensAt(walking(mwt1gt_6),642).
orientation(mwt1gt_5,150,643). coords(mwt1gt_5,82,71,643).  visible(mwt1gt_5,643). happensAt(walking(mwt1gt_5),643). orientation(mwt1gt_6,150,643). coords(mwt1gt_6,74,78,643).  visible(mwt1gt_6,643). happensAt(walking(mwt1gt_6),643).
orientation(mwt1gt_5,150,644). coords(mwt1gt_5,82,71,644).  visible(mwt1gt_5,644). happensAt(walking(mwt1gt_5),644). orientation(mwt1gt_6,150,644). coords(mwt1gt_6,74,78,644).  visible(mwt1gt_6,644). happensAt(walking(mwt1gt_6),644).
orientation(mwt1gt_5,150,645). coords(mwt1gt_5,82,71,645).  visible(mwt1gt_5,645). happensAt(walking(mwt1gt_5),645). orientation(mwt1gt_6,150,645). coords(mwt1gt_6,74,78,645).  visible(mwt1gt_6,645). happensAt(walking(mwt1gt_6),645).
orientation(mwt1gt_5,150,646). coords(mwt1gt_5,82,71,646).  visible(mwt1gt_5,646). happensAt(walking(mwt1gt_5),646). orientation(mwt1gt_6,150,646). coords(mwt1gt_6,74,78,646).  visible(mwt1gt_6,646). happensAt(walking(mwt1gt_6),646).
orientation(mwt1gt_5,150,647). coords(mwt1gt_5,82,71,647).  visible(mwt1gt_5,647). happensAt(walking(mwt1gt_5),647). orientation(mwt1gt_6,150,647). coords(mwt1gt_6,74,78,647).  visible(mwt1gt_6,647). happensAt(walking(mwt1gt_6),647).
orientation(mwt1gt_5,150,648). coords(mwt1gt_5,82,75,648).  visible(mwt1gt_5,648). happensAt(walking(mwt1gt_5),648). orientation(mwt1gt_6,150,648). coords(mwt1gt_6,74,82,648).  visible(mwt1gt_6,648). happensAt(walking(mwt1gt_6),648).
orientation(mwt1gt_5,150,649). coords(mwt1gt_5,82,75,649).  visible(mwt1gt_5,649). happensAt(walking(mwt1gt_5),649). orientation(mwt1gt_6,150,649). coords(mwt1gt_6,74,82,649).  visible(mwt1gt_6,649). happensAt(walking(mwt1gt_6),649).
orientation(mwt1gt_5,150,650). coords(mwt1gt_5,82,75,650).  visible(mwt1gt_5,650). happensAt(walking(mwt1gt_5),650). orientation(mwt1gt_6,150,650). coords(mwt1gt_6,74,82,650).  visible(mwt1gt_6,650). happensAt(walking(mwt1gt_6),650).
orientation(mwt1gt_5,150,651). coords(mwt1gt_5,82,75,651).  visible(mwt1gt_5,651). happensAt(walking(mwt1gt_5),651). orientation(mwt1gt_6,150,651). coords(mwt1gt_6,74,82,651).  visible(mwt1gt_6,651). happensAt(walking(mwt1gt_6),651).
orientation(mwt1gt_5,150,652). coords(mwt1gt_5,82,75,652).  visible(mwt1gt_5,652). happensAt(walking(mwt1gt_5),652). orientation(mwt1gt_6,150,652). coords(mwt1gt_6,74,82,652).  visible(mwt1gt_6,652). happensAt(walking(mwt1gt_6),652).
orientation(mwt1gt_5,150,653). coords(mwt1gt_5,82,75,653).  visible(mwt1gt_5,653). happensAt(walking(mwt1gt_5),653). orientation(mwt1gt_6,150,653). coords(mwt1gt_6,74,82,653).  visible(mwt1gt_6,653). happensAt(walking(mwt1gt_6),653).
orientation(mwt1gt_5,150,654). coords(mwt1gt_5,83,78,654).  visible(mwt1gt_5,654). happensAt(walking(mwt1gt_5),654). orientation(mwt1gt_6,150,654). coords(mwt1gt_6,74,82,654).  visible(mwt1gt_6,654). happensAt(walking(mwt1gt_6),654).
orientation(mwt1gt_5,150,655). coords(mwt1gt_5,83,78,655).  visible(mwt1gt_5,655). happensAt(walking(mwt1gt_5),655). orientation(mwt1gt_6,150,655). coords(mwt1gt_6,74,82,655).  visible(mwt1gt_6,655). happensAt(walking(mwt1gt_6),655).
orientation(mwt1gt_5,150,656). coords(mwt1gt_5,83,78,656).  visible(mwt1gt_5,656). happensAt(walking(mwt1gt_5),656). orientation(mwt1gt_6,150,656). coords(mwt1gt_6,74,82,656).  visible(mwt1gt_6,656). happensAt(walking(mwt1gt_6),656).
orientation(mwt1gt_5,150,657). coords(mwt1gt_5,83,78,657).  visible(mwt1gt_5,657). happensAt(walking(mwt1gt_5),657). orientation(mwt1gt_6,150,657). coords(mwt1gt_6,75,83,657).  visible(mwt1gt_6,657). happensAt(walking(mwt1gt_6),657).
orientation(mwt1gt_5,150,658). coords(mwt1gt_5,83,78,658).  visible(mwt1gt_5,658). happensAt(walking(mwt1gt_5),658). orientation(mwt1gt_6,150,658). coords(mwt1gt_6,75,83,658).  visible(mwt1gt_6,658). happensAt(walking(mwt1gt_6),658).
orientation(mwt1gt_5,150,659). coords(mwt1gt_5,84,80,659).  visible(mwt1gt_5,659). happensAt(walking(mwt1gt_5),659). orientation(mwt1gt_6,150,659). coords(mwt1gt_6,75,84,659).  visible(mwt1gt_6,659). happensAt(walking(mwt1gt_6),659).
orientation(mwt1gt_5,150,660). coords(mwt1gt_5,84,80,660).  visible(mwt1gt_5,660). happensAt(walking(mwt1gt_5),660). orientation(mwt1gt_6,150,660). coords(mwt1gt_6,75,84,660).  visible(mwt1gt_6,660). happensAt(walking(mwt1gt_6),660).
orientation(mwt1gt_5,150,661). coords(mwt1gt_5,84,80,661).  visible(mwt1gt_5,661). happensAt(walking(mwt1gt_5),661). orientation(mwt1gt_6,150,661). coords(mwt1gt_6,75,84,661).  visible(mwt1gt_6,661). happensAt(walking(mwt1gt_6),661).
#show holdsAt/2.