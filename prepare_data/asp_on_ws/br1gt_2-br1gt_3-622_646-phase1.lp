
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

	orientation(br1gt_2,12,622). coords(br1gt_2,86,209,622).  visible(br1gt_2,622). happensAt(inactive(br1gt_2),622). orientation(br1gt_3,113,622). coords(br1gt_3,298,96,622).  visible(br1gt_3,622). happensAt(walking(br1gt_3),622).
orientation(br1gt_2,12,623). coords(br1gt_2,86,209,623).  visible(br1gt_2,623). happensAt(inactive(br1gt_2),623). orientation(br1gt_3,113,623). coords(br1gt_3,298,96,623).  visible(br1gt_3,623). happensAt(active(br1gt_3),623).
orientation(br1gt_2,12,624). coords(br1gt_2,86,209,624).  visible(br1gt_2,624). happensAt(inactive(br1gt_2),624). orientation(br1gt_3,108,624). coords(br1gt_3,298,96,624).  visible(br1gt_3,624). happensAt(active(br1gt_3),624).
orientation(br1gt_2,12,625). coords(br1gt_2,86,209,625).  visible(br1gt_2,625). happensAt(inactive(br1gt_2),625). orientation(br1gt_3,108,625). coords(br1gt_3,299,96,625).  visible(br1gt_3,625). happensAt(active(br1gt_3),625).
orientation(br1gt_2,12,626). coords(br1gt_2,86,209,626).  visible(br1gt_2,626). happensAt(inactive(br1gt_2),626). orientation(br1gt_3,108,626). coords(br1gt_3,299,95,626).  visible(br1gt_3,626). happensAt(active(br1gt_3),626).
orientation(br1gt_2,12,627). coords(br1gt_2,86,209,627).  visible(br1gt_2,627). happensAt(inactive(br1gt_2),627). orientation(br1gt_3,108,627). coords(br1gt_3,299,95,627).  visible(br1gt_3,627). happensAt(active(br1gt_3),627).
orientation(br1gt_2,12,628). coords(br1gt_2,86,209,628).  visible(br1gt_2,628). happensAt(inactive(br1gt_2),628). orientation(br1gt_3,108,628). coords(br1gt_3,299,95,628).  visible(br1gt_3,628). happensAt(active(br1gt_3),628).
orientation(br1gt_2,12,629). coords(br1gt_2,86,209,629).  visible(br1gt_2,629). happensAt(inactive(br1gt_2),629). orientation(br1gt_3,108,629). coords(br1gt_3,299,95,629).  visible(br1gt_3,629). happensAt(active(br1gt_3),629).
orientation(br1gt_2,12,630). coords(br1gt_2,86,209,630).  visible(br1gt_2,630). happensAt(inactive(br1gt_2),630). orientation(br1gt_3,108,630). coords(br1gt_3,299,95,630).  visible(br1gt_3,630). happensAt(active(br1gt_3),630).
orientation(br1gt_2,12,631). coords(br1gt_2,86,209,631).  visible(br1gt_2,631). happensAt(inactive(br1gt_2),631). orientation(br1gt_3,108,631). coords(br1gt_3,300,96,631).  visible(br1gt_3,631). happensAt(active(br1gt_3),631).
orientation(br1gt_2,12,632). coords(br1gt_2,86,209,632).  visible(br1gt_2,632). happensAt(inactive(br1gt_2),632). orientation(br1gt_3,108,632). coords(br1gt_3,301,96,632).  visible(br1gt_3,632). happensAt(active(br1gt_3),632).
orientation(br1gt_2,12,633). coords(br1gt_2,86,209,633).  visible(br1gt_2,633). happensAt(inactive(br1gt_2),633). orientation(br1gt_3,108,633). coords(br1gt_3,301,96,633).  visible(br1gt_3,633). happensAt(active(br1gt_3),633).
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
#show holdsAt/2.