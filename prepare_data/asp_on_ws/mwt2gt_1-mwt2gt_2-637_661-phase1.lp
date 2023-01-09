
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

	orientation(mwt2gt_1,153,637). coords(mwt2gt_1,27,148,637).  visible(mwt2gt_1,637). happensAt(walking(mwt2gt_1),637). orientation(mwt2gt_2,152,637). coords(mwt2gt_2,26,148,637).  visible(mwt2gt_2,637). happensAt(walking(mwt2gt_2),637).
orientation(mwt2gt_1,153,638). coords(mwt2gt_1,27,146,638).  visible(mwt2gt_1,638). happensAt(walking(mwt2gt_1),638). orientation(mwt2gt_2,152,638). coords(mwt2gt_2,26,147,638).  visible(mwt2gt_2,638). happensAt(walking(mwt2gt_2),638).
orientation(mwt2gt_1,153,639). coords(mwt2gt_1,27,145,639).  visible(mwt2gt_1,639). happensAt(walking(mwt2gt_1),639). orientation(mwt2gt_2,152,639). coords(mwt2gt_2,26,146,639).  visible(mwt2gt_2,639). happensAt(walking(mwt2gt_2),639).
orientation(mwt2gt_1,153,640). coords(mwt2gt_1,22,145,640).  visible(mwt2gt_1,640). happensAt(walking(mwt2gt_1),640). orientation(mwt2gt_2,152,640). coords(mwt2gt_2,21,146,640).  visible(mwt2gt_2,640). happensAt(walking(mwt2gt_2),640).
orientation(mwt2gt_1,153,641). coords(mwt2gt_1,22,143,641).  visible(mwt2gt_1,641). happensAt(walking(mwt2gt_1),641). orientation(mwt2gt_2,152,641). coords(mwt2gt_2,21,145,641).  visible(mwt2gt_2,641). happensAt(walking(mwt2gt_2),641).
orientation(mwt2gt_1,153,642). coords(mwt2gt_1,22,143,642).  visible(mwt2gt_1,642). happensAt(walking(mwt2gt_1),642). orientation(mwt2gt_2,152,642). coords(mwt2gt_2,21,143,642).  visible(mwt2gt_2,642). happensAt(walking(mwt2gt_2),642).
orientation(mwt2gt_1,153,643). coords(mwt2gt_1,22,143,643).  visible(mwt2gt_1,643). happensAt(walking(mwt2gt_1),643). orientation(mwt2gt_2,152,643). coords(mwt2gt_2,21,143,643).  visible(mwt2gt_2,643). happensAt(walking(mwt2gt_2),643).
orientation(mwt2gt_1,153,644). coords(mwt2gt_1,22,142,644).  visible(mwt2gt_1,644). happensAt(walking(mwt2gt_1),644). orientation(mwt2gt_2,152,644). coords(mwt2gt_2,21,143,644).  visible(mwt2gt_2,644). happensAt(walking(mwt2gt_2),644).
orientation(mwt2gt_1,153,645). coords(mwt2gt_1,22,141,645).  visible(mwt2gt_1,645). happensAt(walking(mwt2gt_1),645). orientation(mwt2gt_2,152,645). coords(mwt2gt_2,21,143,645).  visible(mwt2gt_2,645). happensAt(walking(mwt2gt_2),645).
orientation(mwt2gt_1,153,646). coords(mwt2gt_1,22,140,646).  visible(mwt2gt_1,646). happensAt(walking(mwt2gt_1),646). orientation(mwt2gt_2,152,646). coords(mwt2gt_2,22,143,646).  visible(mwt2gt_2,646). happensAt(walking(mwt2gt_2),646).
orientation(mwt2gt_1,153,647). coords(mwt2gt_1,22,140,647).  visible(mwt2gt_1,647). happensAt(walking(mwt2gt_1),647). orientation(mwt2gt_2,152,647). coords(mwt2gt_2,22,143,647).  visible(mwt2gt_2,647). happensAt(walking(mwt2gt_2),647).
orientation(mwt2gt_1,153,648). coords(mwt2gt_1,22,140,648).  visible(mwt2gt_1,648). happensAt(walking(mwt2gt_1),648). orientation(mwt2gt_2,152,648). coords(mwt2gt_2,22,143,648).  visible(mwt2gt_2,648). happensAt(walking(mwt2gt_2),648).
orientation(mwt2gt_1,153,649). coords(mwt2gt_1,22,139,649).  visible(mwt2gt_1,649). happensAt(walking(mwt2gt_1),649). orientation(mwt2gt_2,152,649). coords(mwt2gt_2,22,141,649).  visible(mwt2gt_2,649). happensAt(walking(mwt2gt_2),649).
orientation(mwt2gt_1,153,650). coords(mwt2gt_1,22,139,650).  visible(mwt2gt_1,650). happensAt(walking(mwt2gt_1),650). orientation(mwt2gt_2,152,650). coords(mwt2gt_2,22,141,650).  visible(mwt2gt_2,650). happensAt(walking(mwt2gt_2),650).
orientation(mwt2gt_1,153,651). coords(mwt2gt_1,22,137,651).  visible(mwt2gt_1,651). happensAt(walking(mwt2gt_1),651). orientation(mwt2gt_2,152,651). coords(mwt2gt_2,22,141,651).  visible(mwt2gt_2,651). happensAt(walking(mwt2gt_2),651).
orientation(mwt2gt_1,153,652). coords(mwt2gt_1,22,137,652).  visible(mwt2gt_1,652). happensAt(walking(mwt2gt_1),652). orientation(mwt2gt_2,152,652). coords(mwt2gt_2,22,141,652).  visible(mwt2gt_2,652). happensAt(walking(mwt2gt_2),652).
orientation(mwt2gt_1,153,653). coords(mwt2gt_1,22,136,653).  visible(mwt2gt_1,653). happensAt(walking(mwt2gt_1),653). orientation(mwt2gt_2,152,653). coords(mwt2gt_2,22,141,653).  visible(mwt2gt_2,653). happensAt(walking(mwt2gt_2),653).
orientation(mwt2gt_1,153,654). coords(mwt2gt_1,22,136,654).  visible(mwt2gt_1,654). happensAt(walking(mwt2gt_1),654). orientation(mwt2gt_2,152,654). coords(mwt2gt_2,22,140,654).  visible(mwt2gt_2,654). happensAt(walking(mwt2gt_2),654).
orientation(mwt2gt_1,153,655). coords(mwt2gt_1,22,136,655).  visible(mwt2gt_1,655). happensAt(walking(mwt2gt_1),655). orientation(mwt2gt_2,152,655). coords(mwt2gt_2,22,140,655).  visible(mwt2gt_2,655). happensAt(walking(mwt2gt_2),655).
orientation(mwt2gt_1,153,656). coords(mwt2gt_1,22,135,656).  visible(mwt2gt_1,656). happensAt(walking(mwt2gt_1),656). orientation(mwt2gt_2,152,656). coords(mwt2gt_2,22,140,656).  visible(mwt2gt_2,656). happensAt(walking(mwt2gt_2),656).
orientation(mwt2gt_1,153,657). coords(mwt2gt_1,23,135,657).  visible(mwt2gt_1,657). happensAt(walking(mwt2gt_1),657). orientation(mwt2gt_2,152,657). coords(mwt2gt_2,22,140,657).  visible(mwt2gt_2,657). happensAt(walking(mwt2gt_2),657).
orientation(mwt2gt_1,153,658). coords(mwt2gt_1,23,135,658).  visible(mwt2gt_1,658). happensAt(walking(mwt2gt_1),658). orientation(mwt2gt_2,152,658). coords(mwt2gt_2,22,140,658).  visible(mwt2gt_2,658). happensAt(walking(mwt2gt_2),658).
orientation(mwt2gt_1,153,659). coords(mwt2gt_1,23,134,659).  visible(mwt2gt_1,659). happensAt(walking(mwt2gt_1),659). orientation(mwt2gt_2,152,659). coords(mwt2gt_2,22,140,659).  visible(mwt2gt_2,659). happensAt(walking(mwt2gt_2),659).
orientation(mwt2gt_1,153,660). coords(mwt2gt_1,23,134,660).  visible(mwt2gt_1,660). happensAt(walking(mwt2gt_1),660). orientation(mwt2gt_2,152,660). coords(mwt2gt_2,22,138,660).  visible(mwt2gt_2,660). happensAt(walking(mwt2gt_2),660).
orientation(mwt2gt_1,153,661). coords(mwt2gt_1,23,134,661).  visible(mwt2gt_1,661). happensAt(walking(mwt2gt_1),661). orientation(mwt2gt_2,152,661). coords(mwt2gt_2,22,138,661).  visible(mwt2gt_2,661). happensAt(walking(mwt2gt_2),661).
#show holdsAt/2.