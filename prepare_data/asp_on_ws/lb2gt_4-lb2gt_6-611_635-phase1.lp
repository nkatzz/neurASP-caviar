
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

	orientation(lb2gt_4,130,611). coords(lb2gt_4,152,24,611).  visible(lb2gt_4,611). happensAt(walking(lb2gt_4),611). orientation(lb2gt_6,151,611). coords(lb2gt_6,72,122,611).  visible(lb2gt_6,611). happensAt(inactive(lb2gt_6),611).
orientation(lb2gt_4,130,612). coords(lb2gt_4,153,23,612).  visible(lb2gt_4,612). happensAt(walking(lb2gt_4),612). orientation(lb2gt_6,151,612). coords(lb2gt_6,72,122,612).  visible(lb2gt_6,612). happensAt(inactive(lb2gt_6),612).
orientation(lb2gt_4,130,613). coords(lb2gt_4,153,23,613).  visible(lb2gt_4,613). happensAt(walking(lb2gt_4),613). orientation(lb2gt_6,151,613). coords(lb2gt_6,72,122,613).  visible(lb2gt_6,613). happensAt(inactive(lb2gt_6),613).
orientation(lb2gt_4,130,614). coords(lb2gt_4,154,22,614).  visible(lb2gt_4,614). happensAt(walking(lb2gt_4),614). orientation(lb2gt_6,151,614). coords(lb2gt_6,72,122,614).  visible(lb2gt_6,614). happensAt(inactive(lb2gt_6),614).
orientation(lb2gt_4,130,615). coords(lb2gt_4,155,21,615).  visible(lb2gt_4,615). happensAt(walking(lb2gt_4),615). orientation(lb2gt_6,151,615). coords(lb2gt_6,72,122,615).  visible(lb2gt_6,615). happensAt(inactive(lb2gt_6),615).
orientation(lb2gt_4,130,616). coords(lb2gt_4,155,21,616).  visible(lb2gt_4,616). happensAt(walking(lb2gt_4),616). orientation(lb2gt_6,151,616). coords(lb2gt_6,72,122,616).  visible(lb2gt_6,616). happensAt(inactive(lb2gt_6),616).
orientation(lb2gt_4,130,617). coords(lb2gt_4,155,21,617).  visible(lb2gt_4,617). happensAt(walking(lb2gt_4),617). orientation(lb2gt_6,151,617). coords(lb2gt_6,72,122,617).  visible(lb2gt_6,617). happensAt(inactive(lb2gt_6),617).
orientation(lb2gt_4,130,618). coords(lb2gt_4,155,20,618).  visible(lb2gt_4,618). happensAt(walking(lb2gt_4),618). orientation(lb2gt_6,151,618). coords(lb2gt_6,72,122,618).  visible(lb2gt_6,618). happensAt(inactive(lb2gt_6),618).
orientation(lb2gt_4,130,619). coords(lb2gt_4,157,20,619).  visible(lb2gt_4,619). happensAt(walking(lb2gt_4),619). orientation(lb2gt_6,151,619). coords(lb2gt_6,72,122,619).  visible(lb2gt_6,619). happensAt(inactive(lb2gt_6),619).
orientation(lb2gt_4,130,620). coords(lb2gt_4,158,20,620).  visible(lb2gt_4,620). happensAt(walking(lb2gt_4),620). orientation(lb2gt_6,151,620). coords(lb2gt_6,72,122,620).  visible(lb2gt_6,620). happensAt(inactive(lb2gt_6),620).
orientation(lb2gt_4,130,621). coords(lb2gt_4,158,20,621).  visible(lb2gt_4,621). happensAt(walking(lb2gt_4),621). orientation(lb2gt_6,151,621). coords(lb2gt_6,72,122,621).  visible(lb2gt_6,621). happensAt(inactive(lb2gt_6),621).
orientation(lb2gt_4,130,622). coords(lb2gt_4,159,20,622).  visible(lb2gt_4,622). happensAt(walking(lb2gt_4),622). orientation(lb2gt_6,151,622). coords(lb2gt_6,72,122,622).  visible(lb2gt_6,622). happensAt(inactive(lb2gt_6),622).
orientation(lb2gt_4,130,623). coords(lb2gt_4,160,20,623).  visible(lb2gt_4,623). happensAt(walking(lb2gt_4),623). orientation(lb2gt_6,151,623). coords(lb2gt_6,72,122,623).  visible(lb2gt_6,623). happensAt(inactive(lb2gt_6),623).
orientation(lb2gt_4,130,624). coords(lb2gt_4,160,19,624).  visible(lb2gt_4,624). happensAt(walking(lb2gt_4),624). orientation(lb2gt_6,151,624). coords(lb2gt_6,72,122,624).  visible(lb2gt_6,624). happensAt(inactive(lb2gt_6),624).
orientation(lb2gt_4,130,625). coords(lb2gt_4,162,18,625).  visible(lb2gt_4,625). happensAt(walking(lb2gt_4),625). orientation(lb2gt_6,151,625). coords(lb2gt_6,72,122,625).  visible(lb2gt_6,625). happensAt(inactive(lb2gt_6),625).
orientation(lb2gt_4,130,626). coords(lb2gt_4,163,17,626).  visible(lb2gt_4,626). happensAt(walking(lb2gt_4),626). orientation(lb2gt_6,151,626). coords(lb2gt_6,72,122,626).  visible(lb2gt_6,626). happensAt(inactive(lb2gt_6),626).
orientation(lb2gt_4,130,627). coords(lb2gt_4,163,17,627).  visible(lb2gt_4,627). happensAt(walking(lb2gt_4),627). orientation(lb2gt_6,151,627). coords(lb2gt_6,72,122,627).  visible(lb2gt_6,627). happensAt(inactive(lb2gt_6),627).
orientation(lb2gt_4,130,628). coords(lb2gt_4,164,16,628).  visible(lb2gt_4,628). happensAt(walking(lb2gt_4),628). orientation(lb2gt_6,151,628). coords(lb2gt_6,72,122,628).  visible(lb2gt_6,628). happensAt(inactive(lb2gt_6),628).
orientation(lb2gt_4,130,629). coords(lb2gt_4,164,16,629).  visible(lb2gt_4,629). happensAt(walking(lb2gt_4),629). orientation(lb2gt_6,151,629). coords(lb2gt_6,72,122,629).  visible(lb2gt_6,629). happensAt(inactive(lb2gt_6),629).
orientation(lb2gt_4,130,630). coords(lb2gt_4,165,16,630).  visible(lb2gt_4,630). happensAt(walking(lb2gt_4),630). orientation(lb2gt_6,151,630). coords(lb2gt_6,72,122,630).  visible(lb2gt_6,630). happensAt(inactive(lb2gt_6),630).
orientation(lb2gt_4,130,631). coords(lb2gt_4,165,16,631).  visible(lb2gt_4,631). happensAt(walking(lb2gt_4),631). orientation(lb2gt_6,151,631). coords(lb2gt_6,72,122,631).  visible(lb2gt_6,631). happensAt(inactive(lb2gt_6),631).
orientation(lb2gt_4,130,632). coords(lb2gt_4,165,16,632).  visible(lb2gt_4,632). happensAt(walking(lb2gt_4),632). orientation(lb2gt_6,151,632). coords(lb2gt_6,72,122,632).  visible(lb2gt_6,632). happensAt(inactive(lb2gt_6),632).
orientation(lb2gt_4,130,633). coords(lb2gt_4,165,16,633).  visible(lb2gt_4,633). happensAt(walking(lb2gt_4),633). orientation(lb2gt_6,151,633). coords(lb2gt_6,72,122,633).  visible(lb2gt_6,633). happensAt(inactive(lb2gt_6),633).
orientation(lb2gt_4,130,634). coords(lb2gt_4,166,15,634).  visible(lb2gt_4,634). happensAt(walking(lb2gt_4),634). orientation(lb2gt_6,151,634). coords(lb2gt_6,72,122,634).  visible(lb2gt_6,634). happensAt(inactive(lb2gt_6),634).
orientation(lb2gt_4,130,635). coords(lb2gt_4,166,15,635).  visible(lb2gt_4,635). happensAt(walking(lb2gt_4),635). orientation(lb2gt_6,151,635). coords(lb2gt_6,72,122,635).  visible(lb2gt_6,635). happensAt(inactive(lb2gt_6),635).
#show holdsAt/2.