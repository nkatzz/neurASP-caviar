
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

	orientation(lbpugt_2,132,635). coords(lbpugt_2,182,52,635).  visible(lbpugt_2,635). happensAt(active(lbpugt_2),635). orientation(lbpugt_4,135,635). coords(lbpugt_4,219,34,635).  visible(lbpugt_4,635). happensAt(inactive(lbpugt_4),635).
orientation(lbpugt_2,132,636). coords(lbpugt_2,182,53,636).  visible(lbpugt_2,636). happensAt(active(lbpugt_2),636). orientation(lbpugt_4,135,636). coords(lbpugt_4,219,34,636).  visible(lbpugt_4,636). happensAt(inactive(lbpugt_4),636).
orientation(lbpugt_2,132,637). coords(lbpugt_2,182,53,637).  visible(lbpugt_2,637). happensAt(active(lbpugt_2),637). orientation(lbpugt_4,135,637). coords(lbpugt_4,219,34,637).  visible(lbpugt_4,637). happensAt(inactive(lbpugt_4),637).
orientation(lbpugt_2,132,638). coords(lbpugt_2,182,54,638).  visible(lbpugt_2,638). happensAt(active(lbpugt_2),638). orientation(lbpugt_4,135,638). coords(lbpugt_4,219,34,638).  visible(lbpugt_4,638). happensAt(inactive(lbpugt_4),638).
orientation(lbpugt_2,132,639). coords(lbpugt_2,183,54,639).  visible(lbpugt_2,639). happensAt(active(lbpugt_2),639). orientation(lbpugt_4,135,639). coords(lbpugt_4,219,34,639).  visible(lbpugt_4,639). happensAt(inactive(lbpugt_4),639).
orientation(lbpugt_2,132,640). coords(lbpugt_2,183,55,640).  visible(lbpugt_2,640). happensAt(active(lbpugt_2),640). orientation(lbpugt_4,135,640). coords(lbpugt_4,219,34,640).  visible(lbpugt_4,640). happensAt(inactive(lbpugt_4),640).
orientation(lbpugt_2,132,641). coords(lbpugt_2,183,55,641).  visible(lbpugt_2,641). happensAt(active(lbpugt_2),641). orientation(lbpugt_4,135,641). coords(lbpugt_4,219,34,641).  visible(lbpugt_4,641). happensAt(inactive(lbpugt_4),641).
orientation(lbpugt_2,132,642). coords(lbpugt_2,184,55,642).  visible(lbpugt_2,642). happensAt(active(lbpugt_2),642). orientation(lbpugt_4,135,642). coords(lbpugt_4,219,34,642).  visible(lbpugt_4,642). happensAt(inactive(lbpugt_4),642).
orientation(lbpugt_2,132,643). coords(lbpugt_2,184,56,643).  visible(lbpugt_2,643). happensAt(walking(lbpugt_2),643). orientation(lbpugt_4,135,643). coords(lbpugt_4,219,34,643).  visible(lbpugt_4,643). happensAt(inactive(lbpugt_4),643).
orientation(lbpugt_2,132,644). coords(lbpugt_2,184,56,644).  visible(lbpugt_2,644). happensAt(walking(lbpugt_2),644). orientation(lbpugt_4,135,644). coords(lbpugt_4,219,34,644).  visible(lbpugt_4,644). happensAt(inactive(lbpugt_4),644).
orientation(lbpugt_2,132,645). coords(lbpugt_2,184,57,645).  visible(lbpugt_2,645). happensAt(walking(lbpugt_2),645). orientation(lbpugt_4,135,645). coords(lbpugt_4,219,34,645).  visible(lbpugt_4,645). happensAt(inactive(lbpugt_4),645).
orientation(lbpugt_2,132,646). coords(lbpugt_2,184,57,646).  visible(lbpugt_2,646). happensAt(walking(lbpugt_2),646). orientation(lbpugt_4,135,646). coords(lbpugt_4,219,34,646).  visible(lbpugt_4,646). happensAt(inactive(lbpugt_4),646).
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
#show holdsAt/2.