
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

	orientation(lbgt_3,61,607). coords(lbgt_3,146,162,607).  visible(lbgt_3,607). happensAt(inactive(lbgt_3),607). orientation(lbgt_4,161,607). coords(lbgt_4,116,154,607).  visible(lbgt_4,607). happensAt(active(lbgt_4),607).
orientation(lbgt_3,61,608). coords(lbgt_3,146,162,608).  visible(lbgt_3,608). happensAt(inactive(lbgt_3),608). orientation(lbgt_4,161,608). coords(lbgt_4,116,154,608).  visible(lbgt_4,608). happensAt(active(lbgt_4),608).
orientation(lbgt_3,61,609). coords(lbgt_3,146,162,609).  visible(lbgt_3,609). happensAt(inactive(lbgt_3),609). orientation(lbgt_4,161,609). coords(lbgt_4,116,155,609).  visible(lbgt_4,609). happensAt(active(lbgt_4),609).
orientation(lbgt_3,61,610). coords(lbgt_3,146,162,610).  visible(lbgt_3,610). happensAt(inactive(lbgt_3),610). orientation(lbgt_4,163,610). coords(lbgt_4,116,155,610).  visible(lbgt_4,610). happensAt(active(lbgt_4),610).
orientation(lbgt_3,61,611). coords(lbgt_3,146,162,611).  visible(lbgt_3,611). happensAt(inactive(lbgt_3),611). orientation(lbgt_4,164,611). coords(lbgt_4,116,155,611).  visible(lbgt_4,611). happensAt(walking(lbgt_4),611).
orientation(lbgt_3,61,612). coords(lbgt_3,146,162,612).  visible(lbgt_3,612). happensAt(inactive(lbgt_3),612). orientation(lbgt_4,164,612). coords(lbgt_4,115,155,612).  visible(lbgt_4,612). happensAt(walking(lbgt_4),612).
orientation(lbgt_3,61,613). coords(lbgt_3,146,162,613).  visible(lbgt_3,613). happensAt(inactive(lbgt_3),613). orientation(lbgt_4,165,613). coords(lbgt_4,115,156,613).  visible(lbgt_4,613). happensAt(walking(lbgt_4),613).
orientation(lbgt_3,61,614). coords(lbgt_3,146,162,614).  visible(lbgt_3,614). happensAt(inactive(lbgt_3),614). orientation(lbgt_4,170,614). coords(lbgt_4,114,156,614).  visible(lbgt_4,614). happensAt(walking(lbgt_4),614).
orientation(lbgt_3,61,615). coords(lbgt_3,146,162,615).  visible(lbgt_3,615). happensAt(inactive(lbgt_3),615). orientation(lbgt_4,166,615). coords(lbgt_4,114,158,615).  visible(lbgt_4,615). happensAt(walking(lbgt_4),615).
orientation(lbgt_3,61,616). coords(lbgt_3,146,162,616).  visible(lbgt_3,616). happensAt(inactive(lbgt_3),616). orientation(lbgt_4,164,616). coords(lbgt_4,114,159,616).  visible(lbgt_4,616). happensAt(walking(lbgt_4),616).
orientation(lbgt_3,61,617). coords(lbgt_3,146,162,617).  visible(lbgt_3,617). happensAt(inactive(lbgt_3),617). orientation(lbgt_4,157,617). coords(lbgt_4,114,160,617).  visible(lbgt_4,617). happensAt(walking(lbgt_4),617).
orientation(lbgt_3,61,618). coords(lbgt_3,146,162,618).  visible(lbgt_3,618). happensAt(inactive(lbgt_3),618). orientation(lbgt_4,163,618). coords(lbgt_4,114,161,618).  visible(lbgt_4,618). happensAt(walking(lbgt_4),618).
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
#show holdsAt/2.