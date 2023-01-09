
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

	orientation(br3gt_1,131,630). coords(br3gt_1,184,46,630).  visible(br3gt_1,630). happensAt(active(br3gt_1),630). orientation(br3gt_2,150,630). coords(br3gt_2,2,95,630).  visible(br3gt_2,630). happensAt(walking(br3gt_2),630).
orientation(br3gt_1,131,631). coords(br3gt_1,184,46,631).  visible(br3gt_1,631). happensAt(active(br3gt_1),631). orientation(br3gt_2,150,631). coords(br3gt_2,3,93,631).  visible(br3gt_2,631). happensAt(walking(br3gt_2),631).
orientation(br3gt_1,131,632). coords(br3gt_1,184,46,632).  visible(br3gt_1,632). happensAt(active(br3gt_1),632). orientation(br3gt_2,150,632). coords(br3gt_2,3,93,632).  visible(br3gt_2,632). happensAt(walking(br3gt_2),632).
orientation(br3gt_1,131,633). coords(br3gt_1,184,46,633).  visible(br3gt_1,633). happensAt(active(br3gt_1),633). orientation(br3gt_2,150,633). coords(br3gt_2,3,92,633).  visible(br3gt_2,633). happensAt(walking(br3gt_2),633).
orientation(br3gt_1,131,634). coords(br3gt_1,184,46,634).  visible(br3gt_1,634). happensAt(active(br3gt_1),634). orientation(br3gt_2,150,634). coords(br3gt_2,3,92,634).  visible(br3gt_2,634). happensAt(walking(br3gt_2),634).
orientation(br3gt_1,131,635). coords(br3gt_1,184,46,635).  visible(br3gt_1,635). happensAt(active(br3gt_1),635). orientation(br3gt_2,150,635). coords(br3gt_2,3,92,635).  visible(br3gt_2,635). happensAt(walking(br3gt_2),635).
orientation(br3gt_1,131,636). coords(br3gt_1,184,46,636).  visible(br3gt_1,636). happensAt(active(br3gt_1),636). orientation(br3gt_2,150,636). coords(br3gt_2,3,92,636).  visible(br3gt_2,636). happensAt(walking(br3gt_2),636).
orientation(br3gt_1,131,637). coords(br3gt_1,184,46,637).  visible(br3gt_1,637). happensAt(active(br3gt_1),637). orientation(br3gt_2,150,637). coords(br3gt_2,3,92,637).  visible(br3gt_2,637). happensAt(walking(br3gt_2),637).
orientation(br3gt_1,131,638). coords(br3gt_1,184,46,638).  visible(br3gt_1,638). happensAt(active(br3gt_1),638). orientation(br3gt_2,145,638). coords(br3gt_2,3,92,638).  visible(br3gt_2,638). happensAt(walking(br3gt_2),638).
orientation(br3gt_1,131,639). coords(br3gt_1,184,46,639).  visible(br3gt_1,639). happensAt(active(br3gt_1),639). orientation(br3gt_2,145,639). coords(br3gt_2,3,91,639).  visible(br3gt_2,639). happensAt(walking(br3gt_2),639).
orientation(br3gt_1,131,640). coords(br3gt_1,184,46,640).  visible(br3gt_1,640). happensAt(active(br3gt_1),640). orientation(br3gt_2,145,640). coords(br3gt_2,3,90,640).  visible(br3gt_2,640). happensAt(walking(br3gt_2),640).
orientation(br3gt_1,131,641). coords(br3gt_1,184,46,641).  visible(br3gt_1,641). happensAt(active(br3gt_1),641). orientation(br3gt_2,145,641). coords(br3gt_2,3,88,641).  visible(br3gt_2,641). happensAt(walking(br3gt_2),641).
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
#show holdsAt/2.