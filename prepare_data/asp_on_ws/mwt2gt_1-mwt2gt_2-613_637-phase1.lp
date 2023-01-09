
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

	orientation(mwt2gt_1,153,613). coords(mwt2gt_1,27,155,613).  visible(mwt2gt_1,613). happensAt(walking(mwt2gt_1),613). orientation(mwt2gt_2,152,613). coords(mwt2gt_2,26,162,613).  visible(mwt2gt_2,613). happensAt(walking(mwt2gt_2),613).
orientation(mwt2gt_1,153,614). coords(mwt2gt_1,27,155,614).  visible(mwt2gt_1,614). happensAt(walking(mwt2gt_1),614). orientation(mwt2gt_2,152,614). coords(mwt2gt_2,26,160,614).  visible(mwt2gt_2,614). happensAt(walking(mwt2gt_2),614).
orientation(mwt2gt_1,153,615). coords(mwt2gt_1,27,153,615).  visible(mwt2gt_1,615). happensAt(walking(mwt2gt_1),615). orientation(mwt2gt_2,152,615). coords(mwt2gt_2,26,159,615).  visible(mwt2gt_2,615). happensAt(walking(mwt2gt_2),615).
orientation(mwt2gt_1,153,616). coords(mwt2gt_1,27,154,616).  visible(mwt2gt_1,616). happensAt(walking(mwt2gt_1),616). orientation(mwt2gt_2,152,616). coords(mwt2gt_2,26,157,616).  visible(mwt2gt_2,616). happensAt(walking(mwt2gt_2),616).
orientation(mwt2gt_1,153,617). coords(mwt2gt_1,28,154,617).  visible(mwt2gt_1,617). happensAt(walking(mwt2gt_1),617). orientation(mwt2gt_2,152,617). coords(mwt2gt_2,25,156,617).  visible(mwt2gt_2,617). happensAt(walking(mwt2gt_2),617).
orientation(mwt2gt_1,153,618). coords(mwt2gt_1,28,152,618).  visible(mwt2gt_1,618). happensAt(walking(mwt2gt_1),618). orientation(mwt2gt_2,152,618). coords(mwt2gt_2,25,155,618).  visible(mwt2gt_2,618). happensAt(walking(mwt2gt_2),618).
orientation(mwt2gt_1,153,619). coords(mwt2gt_1,27,152,619).  visible(mwt2gt_1,619). happensAt(walking(mwt2gt_1),619). orientation(mwt2gt_2,152,619). coords(mwt2gt_2,25,155,619).  visible(mwt2gt_2,619). happensAt(walking(mwt2gt_2),619).
orientation(mwt2gt_1,153,620). coords(mwt2gt_1,27,152,620).  visible(mwt2gt_1,620). happensAt(walking(mwt2gt_1),620). orientation(mwt2gt_2,152,620). coords(mwt2gt_2,25,155,620).  visible(mwt2gt_2,620). happensAt(walking(mwt2gt_2),620).
orientation(mwt2gt_1,153,621). coords(mwt2gt_1,27,153,621).  visible(mwt2gt_1,621). happensAt(walking(mwt2gt_1),621). orientation(mwt2gt_2,152,621). coords(mwt2gt_2,25,155,621).  visible(mwt2gt_2,621). happensAt(walking(mwt2gt_2),621).
orientation(mwt2gt_1,153,622). coords(mwt2gt_1,27,153,622).  visible(mwt2gt_1,622). happensAt(walking(mwt2gt_1),622). orientation(mwt2gt_2,152,622). coords(mwt2gt_2,25,155,622).  visible(mwt2gt_2,622). happensAt(walking(mwt2gt_2),622).
orientation(mwt2gt_1,153,623). coords(mwt2gt_1,28,153,623).  visible(mwt2gt_1,623). happensAt(walking(mwt2gt_1),623). orientation(mwt2gt_2,152,623). coords(mwt2gt_2,25,155,623).  visible(mwt2gt_2,623). happensAt(walking(mwt2gt_2),623).
orientation(mwt2gt_1,153,624). coords(mwt2gt_1,28,153,624).  visible(mwt2gt_1,624). happensAt(walking(mwt2gt_1),624). orientation(mwt2gt_2,152,624). coords(mwt2gt_2,25,155,624).  visible(mwt2gt_2,624). happensAt(walking(mwt2gt_2),624).
orientation(mwt2gt_1,153,625). coords(mwt2gt_1,28,153,625).  visible(mwt2gt_1,625). happensAt(walking(mwt2gt_1),625). orientation(mwt2gt_2,152,625). coords(mwt2gt_2,25,154,625).  visible(mwt2gt_2,625). happensAt(walking(mwt2gt_2),625).
orientation(mwt2gt_1,153,626). coords(mwt2gt_1,28,152,626).  visible(mwt2gt_1,626). happensAt(walking(mwt2gt_1),626). orientation(mwt2gt_2,152,626). coords(mwt2gt_2,25,153,626).  visible(mwt2gt_2,626). happensAt(walking(mwt2gt_2),626).
orientation(mwt2gt_1,153,627). coords(mwt2gt_1,28,152,627).  visible(mwt2gt_1,627). happensAt(walking(mwt2gt_1),627). orientation(mwt2gt_2,152,627). coords(mwt2gt_2,25,152,627).  visible(mwt2gt_2,627). happensAt(walking(mwt2gt_2),627).
orientation(mwt2gt_1,153,628). coords(mwt2gt_1,28,151,628).  visible(mwt2gt_1,628). happensAt(walking(mwt2gt_1),628). orientation(mwt2gt_2,152,628). coords(mwt2gt_2,25,151,628).  visible(mwt2gt_2,628). happensAt(walking(mwt2gt_2),628).
orientation(mwt2gt_1,153,629). coords(mwt2gt_1,28,151,629).  visible(mwt2gt_1,629). happensAt(walking(mwt2gt_1),629). orientation(mwt2gt_2,152,629). coords(mwt2gt_2,25,151,629).  visible(mwt2gt_2,629). happensAt(walking(mwt2gt_2),629).
orientation(mwt2gt_1,153,630). coords(mwt2gt_1,28,151,630).  visible(mwt2gt_1,630). happensAt(walking(mwt2gt_1),630). orientation(mwt2gt_2,152,630). coords(mwt2gt_2,25,151,630).  visible(mwt2gt_2,630). happensAt(walking(mwt2gt_2),630).
orientation(mwt2gt_1,153,631). coords(mwt2gt_1,28,150,631).  visible(mwt2gt_1,631). happensAt(walking(mwt2gt_1),631). orientation(mwt2gt_2,152,631). coords(mwt2gt_2,25,151,631).  visible(mwt2gt_2,631). happensAt(walking(mwt2gt_2),631).
orientation(mwt2gt_1,153,632). coords(mwt2gt_1,28,150,632).  visible(mwt2gt_1,632). happensAt(walking(mwt2gt_1),632). orientation(mwt2gt_2,152,632). coords(mwt2gt_2,25,150,632).  visible(mwt2gt_2,632). happensAt(walking(mwt2gt_2),632).
orientation(mwt2gt_1,153,633). coords(mwt2gt_1,28,150,633).  visible(mwt2gt_1,633). happensAt(walking(mwt2gt_1),633). orientation(mwt2gt_2,152,633). coords(mwt2gt_2,25,149,633).  visible(mwt2gt_2,633). happensAt(walking(mwt2gt_2),633).
orientation(mwt2gt_1,153,634). coords(mwt2gt_1,28,148,634).  visible(mwt2gt_1,634). happensAt(walking(mwt2gt_1),634). orientation(mwt2gt_2,152,634). coords(mwt2gt_2,25,148,634).  visible(mwt2gt_2,634). happensAt(walking(mwt2gt_2),634).
orientation(mwt2gt_1,153,635). coords(mwt2gt_1,27,148,635).  visible(mwt2gt_1,635). happensAt(walking(mwt2gt_1),635). orientation(mwt2gt_2,152,635). coords(mwt2gt_2,25,148,635).  visible(mwt2gt_2,635). happensAt(walking(mwt2gt_2),635).
orientation(mwt2gt_1,153,636). coords(mwt2gt_1,27,148,636).  visible(mwt2gt_1,636). happensAt(walking(mwt2gt_1),636). orientation(mwt2gt_2,152,636). coords(mwt2gt_2,26,148,636).  visible(mwt2gt_2,636). happensAt(walking(mwt2gt_2),636).
orientation(mwt2gt_1,153,637). coords(mwt2gt_1,27,148,637).  visible(mwt2gt_1,637). happensAt(walking(mwt2gt_1),637). orientation(mwt2gt_2,152,637). coords(mwt2gt_2,26,148,637).  visible(mwt2gt_2,637). happensAt(walking(mwt2gt_2),637).
#show holdsAt/2.