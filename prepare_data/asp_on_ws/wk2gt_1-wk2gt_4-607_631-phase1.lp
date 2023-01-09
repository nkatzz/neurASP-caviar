
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

	orientation(wk2gt_1,125,607). coords(wk2gt_1,115,53,607).  visible(wk2gt_1,607). happensAt(inactive(wk2gt_1),607). orientation(wk2gt_4,161,607). coords(wk2gt_4,32,116,607).  visible(wk2gt_4,607). happensAt(walking(wk2gt_4),607).
orientation(wk2gt_1,125,608). coords(wk2gt_1,115,53,608).  visible(wk2gt_1,608). happensAt(inactive(wk2gt_1),608). orientation(wk2gt_4,161,608). coords(wk2gt_4,31,116,608).  visible(wk2gt_4,608). happensAt(walking(wk2gt_4),608).
orientation(wk2gt_1,125,609). coords(wk2gt_1,115,53,609).  visible(wk2gt_1,609). happensAt(inactive(wk2gt_1),609). orientation(wk2gt_4,161,609). coords(wk2gt_4,31,117,609).  visible(wk2gt_4,609). happensAt(walking(wk2gt_4),609).
orientation(wk2gt_1,125,610). coords(wk2gt_1,114,53,610).  visible(wk2gt_1,610). happensAt(inactive(wk2gt_1),610). orientation(wk2gt_4,161,610). coords(wk2gt_4,31,117,610).  visible(wk2gt_4,610). happensAt(walking(wk2gt_4),610).
orientation(wk2gt_1,125,611). coords(wk2gt_1,114,53,611).  visible(wk2gt_1,611). happensAt(inactive(wk2gt_1),611). orientation(wk2gt_4,161,611). coords(wk2gt_4,31,117,611).  visible(wk2gt_4,611). happensAt(walking(wk2gt_4),611).
orientation(wk2gt_1,125,612). coords(wk2gt_1,114,53,612).  visible(wk2gt_1,612). happensAt(inactive(wk2gt_1),612). orientation(wk2gt_4,161,612). coords(wk2gt_4,31,118,612).  visible(wk2gt_4,612). happensAt(walking(wk2gt_4),612).
orientation(wk2gt_1,125,613). coords(wk2gt_1,114,53,613).  visible(wk2gt_1,613). happensAt(inactive(wk2gt_1),613). orientation(wk2gt_4,161,613). coords(wk2gt_4,31,118,613).  visible(wk2gt_4,613). happensAt(walking(wk2gt_4),613).
orientation(wk2gt_1,125,614). coords(wk2gt_1,114,53,614).  visible(wk2gt_1,614). happensAt(inactive(wk2gt_1),614). orientation(wk2gt_4,161,614). coords(wk2gt_4,31,119,614).  visible(wk2gt_4,614). happensAt(walking(wk2gt_4),614).
orientation(wk2gt_1,125,615). coords(wk2gt_1,114,53,615).  visible(wk2gt_1,615). happensAt(inactive(wk2gt_1),615). orientation(wk2gt_4,161,615). coords(wk2gt_4,31,119,615).  visible(wk2gt_4,615). happensAt(walking(wk2gt_4),615).
orientation(wk2gt_1,125,616). coords(wk2gt_1,114,53,616).  visible(wk2gt_1,616). happensAt(inactive(wk2gt_1),616). orientation(wk2gt_4,161,616). coords(wk2gt_4,31,119,616).  visible(wk2gt_4,616). happensAt(walking(wk2gt_4),616).
orientation(wk2gt_1,125,617). coords(wk2gt_1,115,53,617).  visible(wk2gt_1,617). happensAt(inactive(wk2gt_1),617). orientation(wk2gt_4,161,617). coords(wk2gt_4,31,120,617).  visible(wk2gt_4,617). happensAt(walking(wk2gt_4),617).
orientation(wk2gt_1,125,618). coords(wk2gt_1,115,53,618).  visible(wk2gt_1,618). happensAt(inactive(wk2gt_1),618). orientation(wk2gt_4,161,618). coords(wk2gt_4,31,120,618).  visible(wk2gt_4,618). happensAt(walking(wk2gt_4),618).
orientation(wk2gt_1,125,619). coords(wk2gt_1,115,53,619).  visible(wk2gt_1,619). happensAt(inactive(wk2gt_1),619). orientation(wk2gt_4,161,619). coords(wk2gt_4,30,120,619).  visible(wk2gt_4,619). happensAt(walking(wk2gt_4),619).
orientation(wk2gt_1,125,620). coords(wk2gt_1,115,53,620).  visible(wk2gt_1,620). happensAt(inactive(wk2gt_1),620). orientation(wk2gt_4,161,620). coords(wk2gt_4,30,120,620).  visible(wk2gt_4,620). happensAt(walking(wk2gt_4),620).
orientation(wk2gt_1,125,621). coords(wk2gt_1,115,53,621).  visible(wk2gt_1,621). happensAt(inactive(wk2gt_1),621). orientation(wk2gt_4,161,621). coords(wk2gt_4,30,120,621).  visible(wk2gt_4,621). happensAt(walking(wk2gt_4),621).
orientation(wk2gt_1,125,622). coords(wk2gt_1,115,53,622).  visible(wk2gt_1,622). happensAt(inactive(wk2gt_1),622). orientation(wk2gt_4,161,622). coords(wk2gt_4,30,120,622).  visible(wk2gt_4,622). happensAt(walking(wk2gt_4),622).
orientation(wk2gt_1,125,623). coords(wk2gt_1,115,53,623).  visible(wk2gt_1,623). happensAt(inactive(wk2gt_1),623). orientation(wk2gt_4,161,623). coords(wk2gt_4,29,120,623).  visible(wk2gt_4,623). happensAt(walking(wk2gt_4),623).
orientation(wk2gt_1,125,624). coords(wk2gt_1,115,53,624).  visible(wk2gt_1,624). happensAt(inactive(wk2gt_1),624). orientation(wk2gt_4,161,624). coords(wk2gt_4,29,121,624).  visible(wk2gt_4,624). happensAt(walking(wk2gt_4),624).
orientation(wk2gt_1,125,625). coords(wk2gt_1,115,53,625).  visible(wk2gt_1,625). happensAt(inactive(wk2gt_1),625). orientation(wk2gt_4,161,625). coords(wk2gt_4,28,122,625).  visible(wk2gt_4,625). happensAt(walking(wk2gt_4),625).
orientation(wk2gt_1,125,626). coords(wk2gt_1,115,53,626).  visible(wk2gt_1,626). happensAt(inactive(wk2gt_1),626). orientation(wk2gt_4,161,626). coords(wk2gt_4,28,122,626).  visible(wk2gt_4,626). happensAt(walking(wk2gt_4),626).
orientation(wk2gt_1,125,627). coords(wk2gt_1,115,53,627).  visible(wk2gt_1,627). happensAt(inactive(wk2gt_1),627). orientation(wk2gt_4,161,627). coords(wk2gt_4,28,122,627).  visible(wk2gt_4,627). happensAt(walking(wk2gt_4),627).
orientation(wk2gt_1,125,628). coords(wk2gt_1,115,53,628).  visible(wk2gt_1,628). happensAt(inactive(wk2gt_1),628). orientation(wk2gt_4,161,628). coords(wk2gt_4,28,122,628).  visible(wk2gt_4,628). happensAt(walking(wk2gt_4),628).
orientation(wk2gt_1,125,629). coords(wk2gt_1,115,53,629).  visible(wk2gt_1,629). happensAt(inactive(wk2gt_1),629). orientation(wk2gt_4,161,629). coords(wk2gt_4,28,122,629).  visible(wk2gt_4,629). happensAt(walking(wk2gt_4),629).
orientation(wk2gt_1,125,630). coords(wk2gt_1,115,53,630).  visible(wk2gt_1,630). happensAt(inactive(wk2gt_1),630). orientation(wk2gt_4,161,630). coords(wk2gt_4,28,122,630).  visible(wk2gt_4,630). happensAt(walking(wk2gt_4),630).
orientation(wk2gt_1,125,631). coords(wk2gt_1,115,53,631).  visible(wk2gt_1,631). happensAt(inactive(wk2gt_1),631). orientation(wk2gt_4,161,631). coords(wk2gt_4,28,123,631).  visible(wk2gt_4,631). happensAt(walking(wk2gt_4),631).
#show holdsAt/2.