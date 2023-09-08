
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

	orientation(br1gt_2,12,634). coords(br1gt_2,86,209,634).  visible(br1gt_2,634). happensAt(inactive(br1gt_2),634). orientation(br1gt_3,108,634). coords(br1gt_3,301,96,634).  visible(br1gt_3,634). happensAt(active(br1gt_3),634).
orientation(br1gt_2,12,635). coords(br1gt_2,86,209,635).  visible(br1gt_2,635). happensAt(inactive(br1gt_2),635). orientation(br1gt_3,114,635). coords(br1gt_3,302,95,635).  visible(br1gt_3,635). happensAt(active(br1gt_3),635).
orientation(br1gt_2,12,636). coords(br1gt_2,86,209,636).  visible(br1gt_2,636). happensAt(inactive(br1gt_2),636). orientation(br1gt_3,114,636). coords(br1gt_3,302,95,636).  visible(br1gt_3,636). happensAt(active(br1gt_3),636).
orientation(br1gt_2,12,637). coords(br1gt_2,86,209,637).  visible(br1gt_2,637). happensAt(inactive(br1gt_2),637). orientation(br1gt_3,114,637). coords(br1gt_3,301,94,637).  visible(br1gt_3,637). happensAt(active(br1gt_3),637).
orientation(br1gt_2,12,638). coords(br1gt_2,86,209,638).  visible(br1gt_2,638). happensAt(inactive(br1gt_2),638). orientation(br1gt_3,110,638). coords(br1gt_3,301,93,638).  visible(br1gt_3,638). happensAt(active(br1gt_3),638).
orientation(br1gt_2,12,639). coords(br1gt_2,86,209,639).  visible(br1gt_2,639). happensAt(inactive(br1gt_2),639). orientation(br1gt_3,110,639). coords(br1gt_3,300,93,639).  visible(br1gt_3,639). happensAt(active(br1gt_3),639).
orientation(br1gt_2,12,640). coords(br1gt_2,86,209,640).  visible(br1gt_2,640). happensAt(inactive(br1gt_2),640). orientation(br1gt_3,110,640). coords(br1gt_3,300,92,640).  visible(br1gt_3,640). happensAt(active(br1gt_3),640).
orientation(br1gt_2,12,641). coords(br1gt_2,86,209,641).  visible(br1gt_2,641). happensAt(inactive(br1gt_2),641). orientation(br1gt_3,110,641). coords(br1gt_3,300,92,641).  visible(br1gt_3,641). happensAt(active(br1gt_3),641).
orientation(br1gt_2,12,642). coords(br1gt_2,86,209,642).  visible(br1gt_2,642). happensAt(inactive(br1gt_2),642). orientation(br1gt_3,110,642). coords(br1gt_3,299,92,642).  visible(br1gt_3,642). happensAt(active(br1gt_3),642).
orientation(br1gt_2,12,643). coords(br1gt_2,86,209,643).  visible(br1gt_2,643). happensAt(inactive(br1gt_2),643). orientation(br1gt_3,110,643). coords(br1gt_3,299,92,643).  visible(br1gt_3,643). happensAt(active(br1gt_3),643).
orientation(br1gt_2,12,644). coords(br1gt_2,86,209,644).  visible(br1gt_2,644). happensAt(inactive(br1gt_2),644). orientation(br1gt_3,110,644). coords(br1gt_3,299,92,644).  visible(br1gt_3,644). happensAt(active(br1gt_3),644).
orientation(br1gt_2,12,645). coords(br1gt_2,86,209,645).  visible(br1gt_2,645). happensAt(inactive(br1gt_2),645). orientation(br1gt_3,110,645). coords(br1gt_3,299,92,645).  visible(br1gt_3,645). happensAt(active(br1gt_3),645).
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
#show holdsAt/2.