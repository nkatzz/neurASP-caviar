
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

	orientation(ms3ggt_0,164,596). coords(ms3ggt_0,79,192,596).  happensAt(appear(ms3ggt_0),596). happensAt(walking(ms3ggt_0),596). orientation(ms3ggt_2,150,596). coords(ms3ggt_2,83,79,596).  visible(ms3ggt_2,596). happensAt(walking(ms3ggt_2),596).
orientation(ms3ggt_0,164,597). coords(ms3ggt_0,78,192,597).  visible(ms3ggt_0,597). happensAt(walking(ms3ggt_0),597). orientation(ms3ggt_2,150,597). coords(ms3ggt_2,83,79,597).  visible(ms3ggt_2,597). happensAt(walking(ms3ggt_2),597).
orientation(ms3ggt_0,164,598). coords(ms3ggt_0,79,192,598).  visible(ms3ggt_0,598). happensAt(walking(ms3ggt_0),598). orientation(ms3ggt_2,150,598). coords(ms3ggt_2,83,79,598).  visible(ms3ggt_2,598). happensAt(walking(ms3ggt_2),598).
orientation(ms3ggt_0,164,599). coords(ms3ggt_0,79,192,599).  visible(ms3ggt_0,599). happensAt(walking(ms3ggt_0),599). orientation(ms3ggt_2,142,599). coords(ms3ggt_2,83,79,599).  visible(ms3ggt_2,599). happensAt(walking(ms3ggt_2),599).
orientation(ms3ggt_0,164,600). coords(ms3ggt_0,79,192,600).  visible(ms3ggt_0,600). happensAt(walking(ms3ggt_0),600). orientation(ms3ggt_2,142,600). coords(ms3ggt_2,83,78,600).  visible(ms3ggt_2,600). happensAt(walking(ms3ggt_2),600).
orientation(ms3ggt_0,164,601). coords(ms3ggt_0,80,192,601).  visible(ms3ggt_0,601). happensAt(walking(ms3ggt_0),601). orientation(ms3ggt_2,142,601). coords(ms3ggt_2,83,78,601).  visible(ms3ggt_2,601). happensAt(walking(ms3ggt_2),601).
orientation(ms3ggt_0,164,602). coords(ms3ggt_0,80,191,602).  visible(ms3ggt_0,602). happensAt(walking(ms3ggt_0),602). orientation(ms3ggt_2,142,602). coords(ms3ggt_2,83,77,602).  visible(ms3ggt_2,602). happensAt(walking(ms3ggt_2),602).
orientation(ms3ggt_0,164,603). coords(ms3ggt_0,83,191,603).  visible(ms3ggt_0,603). happensAt(walking(ms3ggt_0),603). orientation(ms3ggt_2,142,603). coords(ms3ggt_2,83,76,603).  visible(ms3ggt_2,603). happensAt(walking(ms3ggt_2),603).
orientation(ms3ggt_0,160,604). coords(ms3ggt_0,83,191,604).  visible(ms3ggt_0,604). happensAt(walking(ms3ggt_0),604). orientation(ms3ggt_2,142,604). coords(ms3ggt_2,83,76,604).  visible(ms3ggt_2,604). happensAt(walking(ms3ggt_2),604).
orientation(ms3ggt_0,160,605). coords(ms3ggt_0,85,191,605).  visible(ms3ggt_0,605). happensAt(walking(ms3ggt_0),605). orientation(ms3ggt_2,142,605). coords(ms3ggt_2,83,75,605).  visible(ms3ggt_2,605). happensAt(walking(ms3ggt_2),605).
orientation(ms3ggt_0,165,606). coords(ms3ggt_0,86,191,606).  visible(ms3ggt_0,606). happensAt(walking(ms3ggt_0),606). orientation(ms3ggt_2,142,606). coords(ms3ggt_2,83,72,606).  visible(ms3ggt_2,606). happensAt(walking(ms3ggt_2),606).
orientation(ms3ggt_0,165,607). coords(ms3ggt_0,88,191,607).  visible(ms3ggt_0,607). happensAt(walking(ms3ggt_0),607). orientation(ms3ggt_2,142,607). coords(ms3ggt_2,83,71,607).  visible(ms3ggt_2,607). happensAt(walking(ms3ggt_2),607).
orientation(ms3ggt_0,165,608). coords(ms3ggt_0,88,191,608).  visible(ms3ggt_0,608). happensAt(walking(ms3ggt_0),608). orientation(ms3ggt_2,142,608). coords(ms3ggt_2,83,70,608).  visible(ms3ggt_2,608). happensAt(walking(ms3ggt_2),608).
orientation(ms3ggt_0,165,609). coords(ms3ggt_0,89,192,609).  visible(ms3ggt_0,609). happensAt(walking(ms3ggt_0),609). orientation(ms3ggt_2,146,609). coords(ms3ggt_2,83,69,609).  visible(ms3ggt_2,609). happensAt(walking(ms3ggt_2),609).
orientation(ms3ggt_0,165,610). coords(ms3ggt_0,89,192,610).  visible(ms3ggt_0,610). happensAt(walking(ms3ggt_0),610). orientation(ms3ggt_2,146,610). coords(ms3ggt_2,83,69,610).  visible(ms3ggt_2,610). happensAt(walking(ms3ggt_2),610).
orientation(ms3ggt_0,167,611). coords(ms3ggt_0,89,192,611).  visible(ms3ggt_0,611). happensAt(walking(ms3ggt_0),611). orientation(ms3ggt_2,146,611). coords(ms3ggt_2,83,69,611).  visible(ms3ggt_2,611). happensAt(walking(ms3ggt_2),611).
orientation(ms3ggt_0,167,612). coords(ms3ggt_0,89,192,612).  visible(ms3ggt_0,612). happensAt(walking(ms3ggt_0),612). orientation(ms3ggt_2,146,612). coords(ms3ggt_2,83,68,612).  visible(ms3ggt_2,612). happensAt(walking(ms3ggt_2),612).
orientation(ms3ggt_0,167,613). coords(ms3ggt_0,89,192,613).  visible(ms3ggt_0,613). happensAt(walking(ms3ggt_0),613). orientation(ms3ggt_2,146,613). coords(ms3ggt_2,83,68,613).  visible(ms3ggt_2,613). happensAt(walking(ms3ggt_2),613).
orientation(ms3ggt_0,167,614). coords(ms3ggt_0,90,192,614).  visible(ms3ggt_0,614). happensAt(walking(ms3ggt_0),614). orientation(ms3ggt_2,146,614). coords(ms3ggt_2,83,68,614).  visible(ms3ggt_2,614). happensAt(walking(ms3ggt_2),614).
orientation(ms3ggt_0,167,615). coords(ms3ggt_0,91,192,615).  visible(ms3ggt_0,615). happensAt(walking(ms3ggt_0),615). orientation(ms3ggt_2,146,615). coords(ms3ggt_2,83,68,615).  visible(ms3ggt_2,615). happensAt(walking(ms3ggt_2),615).
orientation(ms3ggt_0,167,616). coords(ms3ggt_0,91,192,616).  visible(ms3ggt_0,616). happensAt(walking(ms3ggt_0),616). orientation(ms3ggt_2,146,616). coords(ms3ggt_2,83,66,616).  visible(ms3ggt_2,616). happensAt(walking(ms3ggt_2),616).
orientation(ms3ggt_0,167,617). coords(ms3ggt_0,93,191,617).  visible(ms3ggt_0,617). happensAt(walking(ms3ggt_0),617). orientation(ms3ggt_2,146,617). coords(ms3ggt_2,80,66,617).  visible(ms3ggt_2,617). happensAt(walking(ms3ggt_2),617).
orientation(ms3ggt_0,167,618). coords(ms3ggt_0,96,191,618).  visible(ms3ggt_0,618). happensAt(walking(ms3ggt_0),618). orientation(ms3ggt_2,146,618). coords(ms3ggt_2,80,66,618).  visible(ms3ggt_2,618). happensAt(walking(ms3ggt_2),618).
orientation(ms3ggt_0,167,619). coords(ms3ggt_0,96,191,619).  visible(ms3ggt_0,619). happensAt(walking(ms3ggt_0),619). orientation(ms3ggt_2,146,619). coords(ms3ggt_2,80,66,619).  visible(ms3ggt_2,619). happensAt(walking(ms3ggt_2),619).
orientation(ms3ggt_0,167,620). coords(ms3ggt_0,97,191,620).  visible(ms3ggt_0,620). happensAt(walking(ms3ggt_0),620). orientation(ms3ggt_2,146,620). coords(ms3ggt_2,80,64,620).  visible(ms3ggt_2,620). happensAt(walking(ms3ggt_2),620).
#show holdsAt/2.