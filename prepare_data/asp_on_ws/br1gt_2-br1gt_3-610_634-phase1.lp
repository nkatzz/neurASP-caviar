
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

	orientation(br1gt_2,12,610). coords(br1gt_2,86,209,610).  visible(br1gt_2,610). happensAt(inactive(br1gt_2),610). orientation(br1gt_3,113,610). coords(br1gt_3,290,95,610).  visible(br1gt_3,610). happensAt(walking(br1gt_3),610).
orientation(br1gt_2,12,611). coords(br1gt_2,86,209,611).  visible(br1gt_2,611). happensAt(inactive(br1gt_2),611). orientation(br1gt_3,113,611). coords(br1gt_3,291,95,611).  visible(br1gt_3,611). happensAt(walking(br1gt_3),611).
orientation(br1gt_2,12,612). coords(br1gt_2,86,209,612).  visible(br1gt_2,612). happensAt(inactive(br1gt_2),612). orientation(br1gt_3,113,612). coords(br1gt_3,291,95,612).  visible(br1gt_3,612). happensAt(walking(br1gt_3),612).
orientation(br1gt_2,12,613). coords(br1gt_2,86,209,613).  visible(br1gt_2,613). happensAt(inactive(br1gt_2),613). orientation(br1gt_3,113,613). coords(br1gt_3,292,95,613).  visible(br1gt_3,613). happensAt(walking(br1gt_3),613).
orientation(br1gt_2,12,614). coords(br1gt_2,86,209,614).  visible(br1gt_2,614). happensAt(inactive(br1gt_2),614). orientation(br1gt_3,113,614). coords(br1gt_3,292,95,614).  visible(br1gt_3,614). happensAt(walking(br1gt_3),614).
orientation(br1gt_2,12,615). coords(br1gt_2,86,209,615).  visible(br1gt_2,615). happensAt(inactive(br1gt_2),615). orientation(br1gt_3,113,615). coords(br1gt_3,293,95,615).  visible(br1gt_3,615). happensAt(walking(br1gt_3),615).
orientation(br1gt_2,12,616). coords(br1gt_2,86,209,616).  visible(br1gt_2,616). happensAt(inactive(br1gt_2),616). orientation(br1gt_3,113,616). coords(br1gt_3,293,95,616).  visible(br1gt_3,616). happensAt(walking(br1gt_3),616).
orientation(br1gt_2,12,617). coords(br1gt_2,86,209,617).  visible(br1gt_2,617). happensAt(inactive(br1gt_2),617). orientation(br1gt_3,113,617). coords(br1gt_3,294,95,617).  visible(br1gt_3,617). happensAt(walking(br1gt_3),617).
orientation(br1gt_2,12,618). coords(br1gt_2,86,209,618).  visible(br1gt_2,618). happensAt(inactive(br1gt_2),618). orientation(br1gt_3,113,618). coords(br1gt_3,294,95,618).  visible(br1gt_3,618). happensAt(walking(br1gt_3),618).
orientation(br1gt_2,12,619). coords(br1gt_2,86,209,619).  visible(br1gt_2,619). happensAt(inactive(br1gt_2),619). orientation(br1gt_3,113,619). coords(br1gt_3,295,95,619).  visible(br1gt_3,619). happensAt(walking(br1gt_3),619).
orientation(br1gt_2,12,620). coords(br1gt_2,86,209,620).  visible(br1gt_2,620). happensAt(inactive(br1gt_2),620). orientation(br1gt_3,113,620). coords(br1gt_3,296,95,620).  visible(br1gt_3,620). happensAt(walking(br1gt_3),620).
orientation(br1gt_2,12,621). coords(br1gt_2,86,209,621).  visible(br1gt_2,621). happensAt(inactive(br1gt_2),621). orientation(br1gt_3,113,621). coords(br1gt_3,297,96,621).  visible(br1gt_3,621). happensAt(walking(br1gt_3),621).
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
#show holdsAt/2.