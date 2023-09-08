
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

	orientation(bww1gt_1,173,638). coords(bww1gt_1,102,257,638).  visible(bww1gt_1,638). happensAt(walking(bww1gt_1),638). orientation(bww1gt_2,169,638). coords(bww1gt_2,170,216,638).  visible(bww1gt_2,638). happensAt(walking(bww1gt_2),638).
orientation(bww1gt_1,173,639). coords(bww1gt_1,102,257,639).  visible(bww1gt_1,639). happensAt(walking(bww1gt_1),639). orientation(bww1gt_2,169,639). coords(bww1gt_2,174,217,639).  visible(bww1gt_2,639). happensAt(walking(bww1gt_2),639).
orientation(bww1gt_1,173,640). coords(bww1gt_1,102,257,640).  visible(bww1gt_1,640). happensAt(walking(bww1gt_1),640). orientation(bww1gt_2,169,640). coords(bww1gt_2,178,217,640).  visible(bww1gt_2,640). happensAt(walking(bww1gt_2),640).
orientation(bww1gt_1,173,641). coords(bww1gt_1,102,257,641).  visible(bww1gt_1,641). happensAt(walking(bww1gt_1),641). orientation(bww1gt_2,169,641). coords(bww1gt_2,180,217,641).  visible(bww1gt_2,641). happensAt(walking(bww1gt_2),641).
orientation(bww1gt_1,173,642). coords(bww1gt_1,102,257,642).  visible(bww1gt_1,642). happensAt(walking(bww1gt_1),642). orientation(bww1gt_2,169,642). coords(bww1gt_2,183,216,642).  visible(bww1gt_2,642). happensAt(walking(bww1gt_2),642).
orientation(bww1gt_1,173,643). coords(bww1gt_1,102,257,643).  visible(bww1gt_1,643). happensAt(walking(bww1gt_1),643). orientation(bww1gt_2,169,643). coords(bww1gt_2,185,218,643).  visible(bww1gt_2,643). happensAt(walking(bww1gt_2),643).
orientation(bww1gt_1,173,644). coords(bww1gt_1,102,257,644).  visible(bww1gt_1,644). happensAt(walking(bww1gt_1),644). orientation(bww1gt_2,169,644). coords(bww1gt_2,187,218,644).  visible(bww1gt_2,644). happensAt(walking(bww1gt_2),644).
orientation(bww1gt_1,173,645). coords(bww1gt_1,102,257,645).  visible(bww1gt_1,645). happensAt(walking(bww1gt_1),645). orientation(bww1gt_2,169,645). coords(bww1gt_2,188,219,645).  visible(bww1gt_2,645). happensAt(walking(bww1gt_2),645).
orientation(bww1gt_1,173,646). coords(bww1gt_1,102,257,646).  visible(bww1gt_1,646). happensAt(walking(bww1gt_1),646). orientation(bww1gt_2,169,646). coords(bww1gt_2,189,220,646).  visible(bww1gt_2,646). happensAt(walking(bww1gt_2),646).
orientation(bww1gt_1,173,647). coords(bww1gt_1,102,257,647).  visible(bww1gt_1,647). happensAt(walking(bww1gt_1),647). orientation(bww1gt_2,169,647). coords(bww1gt_2,190,221,647).  visible(bww1gt_2,647). happensAt(walking(bww1gt_2),647).
orientation(bww1gt_1,173,648). coords(bww1gt_1,102,257,648).  visible(bww1gt_1,648). happensAt(walking(bww1gt_1),648). orientation(bww1gt_2,169,648). coords(bww1gt_2,190,223,648).  visible(bww1gt_2,648). happensAt(walking(bww1gt_2),648).
orientation(bww1gt_1,173,649). coords(bww1gt_1,102,257,649).  visible(bww1gt_1,649). happensAt(walking(bww1gt_1),649). orientation(bww1gt_2,169,649). coords(bww1gt_2,191,224,649).  visible(bww1gt_2,649). happensAt(walking(bww1gt_2),649).
orientation(bww1gt_1,173,650). coords(bww1gt_1,102,257,650).  visible(bww1gt_1,650). happensAt(walking(bww1gt_1),650). orientation(bww1gt_2,169,650). coords(bww1gt_2,192,226,650).  visible(bww1gt_2,650). happensAt(walking(bww1gt_2),650).
orientation(bww1gt_1,173,651). coords(bww1gt_1,102,257,651).  visible(bww1gt_1,651). happensAt(walking(bww1gt_1),651). orientation(bww1gt_2,169,651). coords(bww1gt_2,193,228,651).  visible(bww1gt_2,651). happensAt(walking(bww1gt_2),651).
orientation(bww1gt_1,173,652). coords(bww1gt_1,102,257,652).  visible(bww1gt_1,652). happensAt(walking(bww1gt_1),652). orientation(bww1gt_2,169,652). coords(bww1gt_2,193,230,652).  visible(bww1gt_2,652). happensAt(walking(bww1gt_2),652).
orientation(bww1gt_1,173,653). coords(bww1gt_1,102,257,653).  visible(bww1gt_1,653). happensAt(walking(bww1gt_1),653). orientation(bww1gt_2,169,653). coords(bww1gt_2,194,232,653).  visible(bww1gt_2,653). happensAt(walking(bww1gt_2),653).
orientation(bww1gt_1,173,654). coords(bww1gt_1,102,257,654).  visible(bww1gt_1,654). happensAt(walking(bww1gt_1),654). orientation(bww1gt_2,169,654). coords(bww1gt_2,196,235,654).  visible(bww1gt_2,654). happensAt(walking(bww1gt_2),654).
orientation(bww1gt_1,173,655). coords(bww1gt_1,102,257,655).  visible(bww1gt_1,655). happensAt(walking(bww1gt_1),655). orientation(bww1gt_2,169,655). coords(bww1gt_2,198,237,655).  visible(bww1gt_2,655). happensAt(walking(bww1gt_2),655).
orientation(bww1gt_1,173,656). coords(bww1gt_1,102,257,656).  visible(bww1gt_1,656). happensAt(inactive(bww1gt_1),656). orientation(bww1gt_2,169,656). coords(bww1gt_2,199,239,656).  visible(bww1gt_2,656). happensAt(walking(bww1gt_2),656).
orientation(bww1gt_1,173,657). coords(bww1gt_1,102,257,657).  visible(bww1gt_1,657). happensAt(inactive(bww1gt_1),657). orientation(bww1gt_2,169,657). coords(bww1gt_2,200,239,657).  visible(bww1gt_2,657). happensAt(walking(bww1gt_2),657).
orientation(bww1gt_1,173,658). coords(bww1gt_1,102,257,658).  visible(bww1gt_1,658). happensAt(inactive(bww1gt_1),658). orientation(bww1gt_2,169,658). coords(bww1gt_2,201,240,658).  visible(bww1gt_2,658). happensAt(walking(bww1gt_2),658).
orientation(bww1gt_1,173,659). coords(bww1gt_1,102,257,659).  visible(bww1gt_1,659). happensAt(inactive(bww1gt_1),659). orientation(bww1gt_2,169,659). coords(bww1gt_2,203,240,659).  visible(bww1gt_2,659). happensAt(walking(bww1gt_2),659).
orientation(bww1gt_1,173,660). coords(bww1gt_1,102,257,660).  visible(bww1gt_1,660). happensAt(inactive(bww1gt_1),660). orientation(bww1gt_2,169,660). coords(bww1gt_2,205,241,660).  visible(bww1gt_2,660). happensAt(walking(bww1gt_2),660).
orientation(bww1gt_1,173,661). coords(bww1gt_1,102,257,661).  visible(bww1gt_1,661). happensAt(inactive(bww1gt_1),661). orientation(bww1gt_2,169,661). coords(bww1gt_2,205,242,661).  visible(bww1gt_2,661). happensAt(walking(bww1gt_2),661).
orientation(bww1gt_1,173,662). coords(bww1gt_1,102,257,662).  visible(bww1gt_1,662). happensAt(inactive(bww1gt_1),662). orientation(bww1gt_2,169,662). coords(bww1gt_2,205,244,662).  visible(bww1gt_2,662). happensAt(walking(bww1gt_2),662).
#show holdsAt/2.