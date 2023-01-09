
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

	orientation(lbgt_3,61,619). coords(lbgt_3,146,162,619).  visible(lbgt_3,619). happensAt(inactive(lbgt_3),619). orientation(lbgt_4,159,619). coords(lbgt_4,114,162,619).  visible(lbgt_4,619). happensAt(walking(lbgt_4),619).
orientation(lbgt_3,61,620). coords(lbgt_3,146,162,620).  visible(lbgt_3,620). happensAt(inactive(lbgt_3),620). orientation(lbgt_4,158,620). coords(lbgt_4,114,163,620).  visible(lbgt_4,620). happensAt(walking(lbgt_4),620).
orientation(lbgt_3,61,621). coords(lbgt_3,146,162,621).  visible(lbgt_3,621). happensAt(inactive(lbgt_3),621). orientation(lbgt_4,157,621). coords(lbgt_4,114,163,621).  visible(lbgt_4,621). happensAt(walking(lbgt_4),621).
orientation(lbgt_3,61,622). coords(lbgt_3,146,162,622).  visible(lbgt_3,622). happensAt(inactive(lbgt_3),622). orientation(lbgt_4,156,622). coords(lbgt_4,113,164,622).  visible(lbgt_4,622). happensAt(walking(lbgt_4),622).
orientation(lbgt_3,61,623). coords(lbgt_3,146,162,623).  visible(lbgt_3,623). happensAt(inactive(lbgt_3),623). orientation(lbgt_4,160,623). coords(lbgt_4,113,164,623).  visible(lbgt_4,623). happensAt(walking(lbgt_4),623).
orientation(lbgt_3,61,624). coords(lbgt_3,146,162,624).  visible(lbgt_3,624). happensAt(inactive(lbgt_3),624). orientation(lbgt_4,163,624). coords(lbgt_4,113,165,624).  visible(lbgt_4,624). happensAt(walking(lbgt_4),624).
orientation(lbgt_3,61,625). coords(lbgt_3,146,162,625).  visible(lbgt_3,625). happensAt(inactive(lbgt_3),625). orientation(lbgt_4,162,625). coords(lbgt_4,113,165,625).  visible(lbgt_4,625). happensAt(walking(lbgt_4),625).
orientation(lbgt_3,61,626). coords(lbgt_3,146,162,626).  visible(lbgt_3,626). happensAt(inactive(lbgt_3),626). orientation(lbgt_4,163,626). coords(lbgt_4,113,165,626).  visible(lbgt_4,626). happensAt(walking(lbgt_4),626).
orientation(lbgt_3,61,627). coords(lbgt_3,146,162,627).  visible(lbgt_3,627). happensAt(inactive(lbgt_3),627). orientation(lbgt_4,165,627). coords(lbgt_4,113,165,627).  visible(lbgt_4,627). happensAt(walking(lbgt_4),627).
orientation(lbgt_3,61,628). coords(lbgt_3,146,162,628).  visible(lbgt_3,628). happensAt(inactive(lbgt_3),628). orientation(lbgt_4,165,628). coords(lbgt_4,113,166,628).  visible(lbgt_4,628). happensAt(walking(lbgt_4),628).
orientation(lbgt_3,61,629). coords(lbgt_3,146,162,629).  visible(lbgt_3,629). happensAt(inactive(lbgt_3),629). orientation(lbgt_4,163,629). coords(lbgt_4,113,167,629).  visible(lbgt_4,629). happensAt(walking(lbgt_4),629).
orientation(lbgt_3,61,630). coords(lbgt_3,146,162,630).  visible(lbgt_3,630). happensAt(inactive(lbgt_3),630). orientation(lbgt_4,163,630). coords(lbgt_4,113,167,630).  visible(lbgt_4,630). happensAt(walking(lbgt_4),630).
orientation(lbgt_3,61,631). coords(lbgt_3,146,162,631).  visible(lbgt_3,631). happensAt(inactive(lbgt_3),631). orientation(lbgt_4,164,631). coords(lbgt_4,113,167,631).  visible(lbgt_4,631). happensAt(walking(lbgt_4),631).
orientation(lbgt_3,61,632). coords(lbgt_3,146,162,632).  visible(lbgt_3,632). happensAt(inactive(lbgt_3),632). orientation(lbgt_4,160,632). coords(lbgt_4,113,169,632).  visible(lbgt_4,632). happensAt(walking(lbgt_4),632).
orientation(lbgt_3,61,633). coords(lbgt_3,146,162,633).  visible(lbgt_3,633). happensAt(inactive(lbgt_3),633). orientation(lbgt_4,161,633). coords(lbgt_4,113,170,633).  visible(lbgt_4,633). happensAt(walking(lbgt_4),633).
orientation(lbgt_3,61,634). coords(lbgt_3,146,162,634).  visible(lbgt_3,634). happensAt(inactive(lbgt_3),634). orientation(lbgt_4,148,634). coords(lbgt_4,113,172,634).  visible(lbgt_4,634). happensAt(walking(lbgt_4),634).
orientation(lbgt_3,61,635). coords(lbgt_3,146,162,635).  visible(lbgt_3,635). happensAt(inactive(lbgt_3),635). orientation(lbgt_4,148,635). coords(lbgt_4,114,174,635).  visible(lbgt_4,635). happensAt(walking(lbgt_4),635).
orientation(lbgt_3,61,636). coords(lbgt_3,146,162,636).  visible(lbgt_3,636). happensAt(inactive(lbgt_3),636). orientation(lbgt_4,158,636). coords(lbgt_4,114,174,636).  visible(lbgt_4,636). happensAt(walking(lbgt_4),636).
orientation(lbgt_3,61,637). coords(lbgt_3,146,162,637).  visible(lbgt_3,637). happensAt(inactive(lbgt_3),637). orientation(lbgt_4,159,637). coords(lbgt_4,115,175,637).  visible(lbgt_4,637). happensAt(walking(lbgt_4),637).
orientation(lbgt_3,61,638). coords(lbgt_3,146,162,638).  visible(lbgt_3,638). happensAt(inactive(lbgt_3),638). orientation(lbgt_4,161,638). coords(lbgt_4,116,176,638).  visible(lbgt_4,638). happensAt(walking(lbgt_4),638).
orientation(lbgt_3,61,639). coords(lbgt_3,146,162,639).  visible(lbgt_3,639). happensAt(inactive(lbgt_3),639). orientation(lbgt_4,160,639). coords(lbgt_4,116,176,639).  visible(lbgt_4,639). happensAt(walking(lbgt_4),639).
orientation(lbgt_3,61,640). coords(lbgt_3,146,162,640).  visible(lbgt_3,640). happensAt(inactive(lbgt_3),640). orientation(lbgt_4,163,640). coords(lbgt_4,116,177,640).  visible(lbgt_4,640). happensAt(walking(lbgt_4),640).
orientation(lbgt_3,61,641). coords(lbgt_3,146,162,641).  visible(lbgt_3,641). happensAt(inactive(lbgt_3),641). orientation(lbgt_4,159,641). coords(lbgt_4,116,177,641).  visible(lbgt_4,641). happensAt(walking(lbgt_4),641).
orientation(lbgt_3,61,642). coords(lbgt_3,146,162,642).  visible(lbgt_3,642). happensAt(inactive(lbgt_3),642). orientation(lbgt_4,158,642). coords(lbgt_4,116,177,642).  visible(lbgt_4,642). happensAt(walking(lbgt_4),642).
orientation(lbgt_3,61,643). coords(lbgt_3,146,162,643).  visible(lbgt_3,643). happensAt(inactive(lbgt_3),643). orientation(lbgt_4,159,643). coords(lbgt_4,116,177,643).  visible(lbgt_4,643). happensAt(walking(lbgt_4),643).
#show holdsAt/2.