
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

	orientation(br1gt_2,12,598). coords(br1gt_2,86,209,598).  visible(br1gt_2,598). happensAt(inactive(br1gt_2),598). orientation(br1gt_3,129,598). coords(br1gt_3,276,97,598).  visible(br1gt_3,598). happensAt(walking(br1gt_3),598).
orientation(br1gt_2,12,599). coords(br1gt_2,86,209,599).  visible(br1gt_2,599). happensAt(inactive(br1gt_2),599). orientation(br1gt_3,129,599). coords(br1gt_3,277,96,599).  visible(br1gt_3,599). happensAt(walking(br1gt_3),599).
orientation(br1gt_2,12,600). coords(br1gt_2,86,209,600).  visible(br1gt_2,600). happensAt(inactive(br1gt_2),600). orientation(br1gt_3,120,600). coords(br1gt_3,277,96,600).  visible(br1gt_3,600). happensAt(walking(br1gt_3),600).
orientation(br1gt_2,12,601). coords(br1gt_2,86,209,601).  visible(br1gt_2,601). happensAt(inactive(br1gt_2),601). orientation(br1gt_3,120,601). coords(br1gt_3,279,96,601).  visible(br1gt_3,601). happensAt(walking(br1gt_3),601).
orientation(br1gt_2,12,602). coords(br1gt_2,86,209,602).  visible(br1gt_2,602). happensAt(inactive(br1gt_2),602). orientation(br1gt_3,120,602). coords(br1gt_3,279,96,602).  visible(br1gt_3,602). happensAt(walking(br1gt_3),602).
orientation(br1gt_2,12,603). coords(br1gt_2,86,209,603).  visible(br1gt_2,603). happensAt(inactive(br1gt_2),603). orientation(br1gt_3,120,603). coords(br1gt_3,280,95,603).  visible(br1gt_3,603). happensAt(walking(br1gt_3),603).
orientation(br1gt_2,12,604). coords(br1gt_2,86,209,604).  visible(br1gt_2,604). happensAt(inactive(br1gt_2),604). orientation(br1gt_3,120,604). coords(br1gt_3,280,95,604).  visible(br1gt_3,604). happensAt(walking(br1gt_3),604).
orientation(br1gt_2,12,605). coords(br1gt_2,86,209,605).  visible(br1gt_2,605). happensAt(inactive(br1gt_2),605). orientation(br1gt_3,113,605). coords(br1gt_3,281,95,605).  visible(br1gt_3,605). happensAt(walking(br1gt_3),605).
orientation(br1gt_2,12,606). coords(br1gt_2,86,209,606).  visible(br1gt_2,606). happensAt(inactive(br1gt_2),606). orientation(br1gt_3,113,606). coords(br1gt_3,283,95,606).  visible(br1gt_3,606). happensAt(walking(br1gt_3),606).
orientation(br1gt_2,12,607). coords(br1gt_2,86,209,607).  visible(br1gt_2,607). happensAt(inactive(br1gt_2),607). orientation(br1gt_3,113,607). coords(br1gt_3,286,95,607).  visible(br1gt_3,607). happensAt(walking(br1gt_3),607).
orientation(br1gt_2,12,608). coords(br1gt_2,86,209,608).  visible(br1gt_2,608). happensAt(inactive(br1gt_2),608). orientation(br1gt_3,113,608). coords(br1gt_3,289,95,608).  visible(br1gt_3,608). happensAt(walking(br1gt_3),608).
orientation(br1gt_2,12,609). coords(br1gt_2,86,209,609).  visible(br1gt_2,609). happensAt(inactive(br1gt_2),609). orientation(br1gt_3,113,609). coords(br1gt_3,290,95,609).  visible(br1gt_3,609). happensAt(walking(br1gt_3),609).
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
#show holdsAt/2.