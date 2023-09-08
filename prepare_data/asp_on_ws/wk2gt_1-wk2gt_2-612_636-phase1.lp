
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

	orientation(wk2gt_1,125,612). coords(wk2gt_1,114,53,612).  visible(wk2gt_1,612). happensAt(inactive(wk2gt_1),612). orientation(wk2gt_2,142,612). coords(wk2gt_2,99,63,612).  visible(wk2gt_2,612). happensAt(inactive(wk2gt_2),612).
orientation(wk2gt_1,125,613). coords(wk2gt_1,114,53,613).  visible(wk2gt_1,613). happensAt(inactive(wk2gt_1),613). orientation(wk2gt_2,142,613). coords(wk2gt_2,99,63,613).  visible(wk2gt_2,613). happensAt(inactive(wk2gt_2),613).
orientation(wk2gt_1,125,614). coords(wk2gt_1,114,53,614).  visible(wk2gt_1,614). happensAt(inactive(wk2gt_1),614). orientation(wk2gt_2,142,614). coords(wk2gt_2,99,63,614).  visible(wk2gt_2,614). happensAt(inactive(wk2gt_2),614).
orientation(wk2gt_1,125,615). coords(wk2gt_1,114,53,615).  visible(wk2gt_1,615). happensAt(inactive(wk2gt_1),615). orientation(wk2gt_2,142,615). coords(wk2gt_2,99,63,615).  visible(wk2gt_2,615). happensAt(inactive(wk2gt_2),615).
orientation(wk2gt_1,125,616). coords(wk2gt_1,114,53,616).  visible(wk2gt_1,616). happensAt(inactive(wk2gt_1),616). orientation(wk2gt_2,142,616). coords(wk2gt_2,99,63,616).  visible(wk2gt_2,616). happensAt(inactive(wk2gt_2),616).
orientation(wk2gt_1,125,617). coords(wk2gt_1,115,53,617).  visible(wk2gt_1,617). happensAt(inactive(wk2gt_1),617). orientation(wk2gt_2,142,617). coords(wk2gt_2,99,63,617).  visible(wk2gt_2,617). happensAt(inactive(wk2gt_2),617).
orientation(wk2gt_1,125,618). coords(wk2gt_1,115,53,618).  visible(wk2gt_1,618). happensAt(inactive(wk2gt_1),618). orientation(wk2gt_2,142,618). coords(wk2gt_2,99,63,618).  visible(wk2gt_2,618). happensAt(inactive(wk2gt_2),618).
orientation(wk2gt_1,125,619). coords(wk2gt_1,115,53,619).  visible(wk2gt_1,619). happensAt(inactive(wk2gt_1),619). orientation(wk2gt_2,142,619). coords(wk2gt_2,99,63,619).  visible(wk2gt_2,619). happensAt(inactive(wk2gt_2),619).
orientation(wk2gt_1,125,620). coords(wk2gt_1,115,53,620).  visible(wk2gt_1,620). happensAt(inactive(wk2gt_1),620). orientation(wk2gt_2,142,620). coords(wk2gt_2,99,63,620).  visible(wk2gt_2,620). happensAt(inactive(wk2gt_2),620).
orientation(wk2gt_1,125,621). coords(wk2gt_1,115,53,621).  visible(wk2gt_1,621). happensAt(inactive(wk2gt_1),621). orientation(wk2gt_2,142,621). coords(wk2gt_2,99,63,621).  visible(wk2gt_2,621). happensAt(inactive(wk2gt_2),621).
orientation(wk2gt_1,125,622). coords(wk2gt_1,115,53,622).  visible(wk2gt_1,622). happensAt(inactive(wk2gt_1),622). orientation(wk2gt_2,142,622). coords(wk2gt_2,99,63,622).  visible(wk2gt_2,622). happensAt(inactive(wk2gt_2),622).
orientation(wk2gt_1,125,623). coords(wk2gt_1,115,53,623).  visible(wk2gt_1,623). happensAt(inactive(wk2gt_1),623). orientation(wk2gt_2,142,623). coords(wk2gt_2,99,63,623).  visible(wk2gt_2,623). happensAt(inactive(wk2gt_2),623).
orientation(wk2gt_1,125,624). coords(wk2gt_1,115,53,624).  visible(wk2gt_1,624). happensAt(inactive(wk2gt_1),624). orientation(wk2gt_2,142,624). coords(wk2gt_2,99,63,624).  visible(wk2gt_2,624). happensAt(inactive(wk2gt_2),624).
orientation(wk2gt_1,125,625). coords(wk2gt_1,115,53,625).  visible(wk2gt_1,625). happensAt(inactive(wk2gt_1),625). orientation(wk2gt_2,142,625). coords(wk2gt_2,99,63,625).  visible(wk2gt_2,625). happensAt(inactive(wk2gt_2),625).
orientation(wk2gt_1,125,626). coords(wk2gt_1,115,53,626).  visible(wk2gt_1,626). happensAt(inactive(wk2gt_1),626). orientation(wk2gt_2,142,626). coords(wk2gt_2,99,63,626).  visible(wk2gt_2,626). happensAt(inactive(wk2gt_2),626).
orientation(wk2gt_1,125,627). coords(wk2gt_1,115,53,627).  visible(wk2gt_1,627). happensAt(inactive(wk2gt_1),627). orientation(wk2gt_2,142,627). coords(wk2gt_2,99,63,627).  visible(wk2gt_2,627). happensAt(inactive(wk2gt_2),627).
orientation(wk2gt_1,125,628). coords(wk2gt_1,115,53,628).  visible(wk2gt_1,628). happensAt(inactive(wk2gt_1),628). orientation(wk2gt_2,142,628). coords(wk2gt_2,99,63,628).  visible(wk2gt_2,628). happensAt(inactive(wk2gt_2),628).
orientation(wk2gt_1,125,629). coords(wk2gt_1,115,53,629).  visible(wk2gt_1,629). happensAt(inactive(wk2gt_1),629). orientation(wk2gt_2,142,629). coords(wk2gt_2,99,63,629).  visible(wk2gt_2,629). happensAt(inactive(wk2gt_2),629).
orientation(wk2gt_1,125,630). coords(wk2gt_1,115,53,630).  visible(wk2gt_1,630). happensAt(inactive(wk2gt_1),630). orientation(wk2gt_2,142,630). coords(wk2gt_2,99,63,630).  visible(wk2gt_2,630). happensAt(inactive(wk2gt_2),630).
orientation(wk2gt_1,125,631). coords(wk2gt_1,115,53,631).  visible(wk2gt_1,631). happensAt(inactive(wk2gt_1),631). orientation(wk2gt_2,142,631). coords(wk2gt_2,99,63,631).  visible(wk2gt_2,631). happensAt(inactive(wk2gt_2),631).
orientation(wk2gt_1,125,632). coords(wk2gt_1,115,53,632).  visible(wk2gt_1,632). happensAt(inactive(wk2gt_1),632). orientation(wk2gt_2,142,632). coords(wk2gt_2,99,63,632).  visible(wk2gt_2,632). happensAt(inactive(wk2gt_2),632).
orientation(wk2gt_1,125,633). coords(wk2gt_1,115,53,633).  visible(wk2gt_1,633). happensAt(inactive(wk2gt_1),633). orientation(wk2gt_2,142,633). coords(wk2gt_2,99,63,633).  visible(wk2gt_2,633). happensAt(inactive(wk2gt_2),633).
orientation(wk2gt_1,125,634). coords(wk2gt_1,115,53,634).  visible(wk2gt_1,634). happensAt(inactive(wk2gt_1),634). orientation(wk2gt_2,142,634). coords(wk2gt_2,99,63,634).  visible(wk2gt_2,634). happensAt(inactive(wk2gt_2),634).
orientation(wk2gt_1,125,635). coords(wk2gt_1,115,53,635).  visible(wk2gt_1,635). happensAt(inactive(wk2gt_1),635). orientation(wk2gt_2,142,635). coords(wk2gt_2,99,63,635).  visible(wk2gt_2,635). happensAt(inactive(wk2gt_2),635).
orientation(wk2gt_1,125,636). coords(wk2gt_1,115,53,636).  visible(wk2gt_1,636). happensAt(inactive(wk2gt_1),636). orientation(wk2gt_2,142,636). coords(wk2gt_2,99,63,636).  visible(wk2gt_2,636). happensAt(inactive(wk2gt_2),636).
#show holdsAt/2.