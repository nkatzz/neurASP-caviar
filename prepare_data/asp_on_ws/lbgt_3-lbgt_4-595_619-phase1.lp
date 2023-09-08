
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

	orientation(lbgt_3,61,595). coords(lbgt_3,146,162,595).  visible(lbgt_3,595). happensAt(inactive(lbgt_3),595). orientation(lbgt_4,161,595). coords(lbgt_4,117,153,595).  visible(lbgt_4,595). happensAt(active(lbgt_4),595).
orientation(lbgt_3,61,596). coords(lbgt_3,146,162,596).  visible(lbgt_3,596). happensAt(inactive(lbgt_3),596). orientation(lbgt_4,161,596). coords(lbgt_4,117,153,596).  visible(lbgt_4,596). happensAt(active(lbgt_4),596).
orientation(lbgt_3,61,597). coords(lbgt_3,146,162,597).  visible(lbgt_3,597). happensAt(inactive(lbgt_3),597). orientation(lbgt_4,161,597). coords(lbgt_4,117,153,597).  visible(lbgt_4,597). happensAt(active(lbgt_4),597).
orientation(lbgt_3,61,598). coords(lbgt_3,146,162,598).  visible(lbgt_3,598). happensAt(inactive(lbgt_3),598). orientation(lbgt_4,161,598). coords(lbgt_4,117,153,598).  visible(lbgt_4,598). happensAt(active(lbgt_4),598).
orientation(lbgt_3,61,599). coords(lbgt_3,146,162,599).  visible(lbgt_3,599). happensAt(inactive(lbgt_3),599). orientation(lbgt_4,161,599). coords(lbgt_4,117,153,599).  visible(lbgt_4,599). happensAt(active(lbgt_4),599).
orientation(lbgt_3,61,600). coords(lbgt_3,146,162,600).  visible(lbgt_3,600). happensAt(inactive(lbgt_3),600). orientation(lbgt_4,161,600). coords(lbgt_4,117,153,600).  visible(lbgt_4,600). happensAt(active(lbgt_4),600).
orientation(lbgt_3,61,601). coords(lbgt_3,146,162,601).  visible(lbgt_3,601). happensAt(inactive(lbgt_3),601). orientation(lbgt_4,161,601). coords(lbgt_4,117,153,601).  visible(lbgt_4,601). happensAt(active(lbgt_4),601).
orientation(lbgt_3,61,602). coords(lbgt_3,146,162,602).  visible(lbgt_3,602). happensAt(inactive(lbgt_3),602). orientation(lbgt_4,161,602). coords(lbgt_4,117,153,602).  visible(lbgt_4,602). happensAt(active(lbgt_4),602).
orientation(lbgt_3,61,603). coords(lbgt_3,146,162,603).  visible(lbgt_3,603). happensAt(inactive(lbgt_3),603). orientation(lbgt_4,161,603). coords(lbgt_4,117,153,603).  visible(lbgt_4,603). happensAt(active(lbgt_4),603).
orientation(lbgt_3,61,604). coords(lbgt_3,146,162,604).  visible(lbgt_3,604). happensAt(inactive(lbgt_3),604). orientation(lbgt_4,161,604). coords(lbgt_4,117,154,604).  visible(lbgt_4,604). happensAt(active(lbgt_4),604).
orientation(lbgt_3,61,605). coords(lbgt_3,146,162,605).  visible(lbgt_3,605). happensAt(inactive(lbgt_3),605). orientation(lbgt_4,161,605). coords(lbgt_4,116,154,605).  visible(lbgt_4,605). happensAt(active(lbgt_4),605).
orientation(lbgt_3,61,606). coords(lbgt_3,146,162,606).  visible(lbgt_3,606). happensAt(inactive(lbgt_3),606). orientation(lbgt_4,161,606). coords(lbgt_4,116,154,606).  visible(lbgt_4,606). happensAt(active(lbgt_4),606).
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
#show holdsAt/2.