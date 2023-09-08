
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

	orientation(br3gt_1,131,618). coords(br3gt_1,184,46,618).  visible(br3gt_1,618). happensAt(active(br3gt_1),618). orientation(br3gt_2,150,618). coords(br3gt_2,1,102,618).  happensAt(appear(br3gt_2),618). happensAt(walking(br3gt_2),618).
orientation(br3gt_1,131,619). coords(br3gt_1,184,46,619).  visible(br3gt_1,619). happensAt(active(br3gt_1),619). orientation(br3gt_2,150,619). coords(br3gt_2,2,101,619).  visible(br3gt_2,619). happensAt(walking(br3gt_2),619).
orientation(br3gt_1,131,620). coords(br3gt_1,184,46,620).  visible(br3gt_1,620). happensAt(active(br3gt_1),620). orientation(br3gt_2,150,620). coords(br3gt_2,2,101,620).  visible(br3gt_2,620). happensAt(walking(br3gt_2),620).
orientation(br3gt_1,131,621). coords(br3gt_1,184,46,621).  visible(br3gt_1,621). happensAt(active(br3gt_1),621). orientation(br3gt_2,150,621). coords(br3gt_2,2,100,621).  visible(br3gt_2,621). happensAt(walking(br3gt_2),621).
orientation(br3gt_1,131,622). coords(br3gt_1,184,46,622).  visible(br3gt_1,622). happensAt(active(br3gt_1),622). orientation(br3gt_2,150,622). coords(br3gt_2,2,100,622).  visible(br3gt_2,622). happensAt(walking(br3gt_2),622).
orientation(br3gt_1,131,623). coords(br3gt_1,184,46,623).  visible(br3gt_1,623). happensAt(active(br3gt_1),623). orientation(br3gt_2,150,623). coords(br3gt_2,2,99,623).  visible(br3gt_2,623). happensAt(walking(br3gt_2),623).
orientation(br3gt_1,131,624). coords(br3gt_1,184,46,624).  visible(br3gt_1,624). happensAt(active(br3gt_1),624). orientation(br3gt_2,150,624). coords(br3gt_2,2,99,624).  visible(br3gt_2,624). happensAt(walking(br3gt_2),624).
orientation(br3gt_1,131,625). coords(br3gt_1,184,46,625).  visible(br3gt_1,625). happensAt(active(br3gt_1),625). orientation(br3gt_2,150,625). coords(br3gt_2,2,99,625).  visible(br3gt_2,625). happensAt(walking(br3gt_2),625).
orientation(br3gt_1,131,626). coords(br3gt_1,184,46,626).  visible(br3gt_1,626). happensAt(active(br3gt_1),626). orientation(br3gt_2,150,626). coords(br3gt_2,2,99,626).  visible(br3gt_2,626). happensAt(walking(br3gt_2),626).
orientation(br3gt_1,131,627). coords(br3gt_1,184,46,627).  visible(br3gt_1,627). happensAt(active(br3gt_1),627). orientation(br3gt_2,150,627). coords(br3gt_2,2,98,627).  visible(br3gt_2,627). happensAt(walking(br3gt_2),627).
orientation(br3gt_1,131,628). coords(br3gt_1,184,46,628).  visible(br3gt_1,628). happensAt(active(br3gt_1),628). orientation(br3gt_2,150,628). coords(br3gt_2,2,97,628).  visible(br3gt_2,628). happensAt(walking(br3gt_2),628).
orientation(br3gt_1,131,629). coords(br3gt_1,184,46,629).  visible(br3gt_1,629). happensAt(active(br3gt_1),629). orientation(br3gt_2,150,629). coords(br3gt_2,2,95,629).  visible(br3gt_2,629). happensAt(walking(br3gt_2),629).
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
#show holdsAt/2.