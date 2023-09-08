
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

	orientation(lbgt_3,61,751). coords(lbgt_3,146,162,751).  visible(lbgt_3,751). happensAt(inactive(lbgt_3),751). orientation(lbgt_4,153,751). coords(lbgt_4,91,134,751).  visible(lbgt_4,751). happensAt(walking(lbgt_4),751).
orientation(lbgt_3,61,752). coords(lbgt_3,146,162,752).  visible(lbgt_3,752). happensAt(inactive(lbgt_3),752). orientation(lbgt_4,153,752). coords(lbgt_4,90,133,752).  visible(lbgt_4,752). happensAt(walking(lbgt_4),752).
orientation(lbgt_3,61,753). coords(lbgt_3,146,162,753).  visible(lbgt_3,753). happensAt(inactive(lbgt_3),753). orientation(lbgt_4,153,753). coords(lbgt_4,90,133,753).  visible(lbgt_4,753). happensAt(walking(lbgt_4),753).
orientation(lbgt_3,61,754). coords(lbgt_3,146,162,754).  visible(lbgt_3,754). happensAt(inactive(lbgt_3),754). orientation(lbgt_4,153,754). coords(lbgt_4,90,133,754).  visible(lbgt_4,754). happensAt(walking(lbgt_4),754).
orientation(lbgt_3,61,755). coords(lbgt_3,146,162,755).  visible(lbgt_3,755). happensAt(inactive(lbgt_3),755). orientation(lbgt_4,153,755). coords(lbgt_4,89,132,755).  visible(lbgt_4,755). happensAt(walking(lbgt_4),755).
orientation(lbgt_3,61,756). coords(lbgt_3,146,162,756).  visible(lbgt_3,756). happensAt(inactive(lbgt_3),756). orientation(lbgt_4,153,756). coords(lbgt_4,89,132,756).  visible(lbgt_4,756). happensAt(walking(lbgt_4),756).
orientation(lbgt_3,61,757). coords(lbgt_3,146,162,757).  visible(lbgt_3,757). happensAt(inactive(lbgt_3),757). orientation(lbgt_4,153,757). coords(lbgt_4,89,131,757).  visible(lbgt_4,757). happensAt(walking(lbgt_4),757).
orientation(lbgt_3,61,758). coords(lbgt_3,146,162,758).  visible(lbgt_3,758). happensAt(inactive(lbgt_3),758). orientation(lbgt_4,153,758). coords(lbgt_4,89,130,758).  visible(lbgt_4,758). happensAt(walking(lbgt_4),758).
orientation(lbgt_3,61,759). coords(lbgt_3,146,162,759).  visible(lbgt_3,759). happensAt(inactive(lbgt_3),759). orientation(lbgt_4,153,759). coords(lbgt_4,89,129,759).  visible(lbgt_4,759). happensAt(walking(lbgt_4),759).
orientation(lbgt_3,61,760). coords(lbgt_3,146,162,760).  visible(lbgt_3,760). happensAt(inactive(lbgt_3),760). orientation(lbgt_4,153,760). coords(lbgt_4,87,128,760).  visible(lbgt_4,760). happensAt(walking(lbgt_4),760).
orientation(lbgt_3,61,761). coords(lbgt_3,146,162,761).  visible(lbgt_3,761). happensAt(inactive(lbgt_3),761). orientation(lbgt_4,153,761). coords(lbgt_4,87,126,761).  visible(lbgt_4,761). happensAt(walking(lbgt_4),761).
orientation(lbgt_3,61,762). coords(lbgt_3,146,162,762).  visible(lbgt_3,762). happensAt(inactive(lbgt_3),762). orientation(lbgt_4,153,762). coords(lbgt_4,87,125,762).  visible(lbgt_4,762). happensAt(walking(lbgt_4),762).
orientation(lbgt_3,61,763). coords(lbgt_3,146,162,763).  visible(lbgt_3,763). happensAt(inactive(lbgt_3),763). orientation(lbgt_4,153,763). coords(lbgt_4,87,125,763).  visible(lbgt_4,763). happensAt(walking(lbgt_4),763).
orientation(lbgt_3,61,764). coords(lbgt_3,146,162,764).  visible(lbgt_3,764). happensAt(inactive(lbgt_3),764). orientation(lbgt_4,153,764). coords(lbgt_4,86,125,764).  visible(lbgt_4,764). happensAt(walking(lbgt_4),764).
orientation(lbgt_3,61,765). coords(lbgt_3,146,162,765).  visible(lbgt_3,765). happensAt(inactive(lbgt_3),765). orientation(lbgt_4,153,765). coords(lbgt_4,86,125,765).  visible(lbgt_4,765). happensAt(walking(lbgt_4),765).
orientation(lbgt_3,61,766). coords(lbgt_3,146,162,766).  visible(lbgt_3,766). happensAt(inactive(lbgt_3),766). orientation(lbgt_4,153,766). coords(lbgt_4,86,124,766).  visible(lbgt_4,766). happensAt(walking(lbgt_4),766).
orientation(lbgt_3,61,767). coords(lbgt_3,146,162,767).  visible(lbgt_3,767). happensAt(inactive(lbgt_3),767). orientation(lbgt_4,149,767). coords(lbgt_4,86,124,767).  visible(lbgt_4,767). happensAt(walking(lbgt_4),767).
orientation(lbgt_3,61,768). coords(lbgt_3,146,162,768).  visible(lbgt_3,768). happensAt(inactive(lbgt_3),768). orientation(lbgt_4,146,768). coords(lbgt_4,86,124,768).  visible(lbgt_4,768). happensAt(walking(lbgt_4),768).
orientation(lbgt_3,61,769). coords(lbgt_3,146,162,769).  visible(lbgt_3,769). happensAt(inactive(lbgt_3),769). orientation(lbgt_4,145,769). coords(lbgt_4,86,123,769).  visible(lbgt_4,769). happensAt(walking(lbgt_4),769).
orientation(lbgt_3,61,770). coords(lbgt_3,146,162,770).  visible(lbgt_3,770). happensAt(inactive(lbgt_3),770). orientation(lbgt_4,143,770). coords(lbgt_4,86,123,770).  visible(lbgt_4,770). happensAt(walking(lbgt_4),770).
orientation(lbgt_3,61,771). coords(lbgt_3,146,162,771).  visible(lbgt_3,771). happensAt(inactive(lbgt_3),771). orientation(lbgt_4,143,771). coords(lbgt_4,85,121,771).  visible(lbgt_4,771). happensAt(walking(lbgt_4),771).
orientation(lbgt_3,61,772). coords(lbgt_3,146,162,772).  visible(lbgt_3,772). happensAt(inactive(lbgt_3),772). orientation(lbgt_4,141,772). coords(lbgt_4,85,120,772).  visible(lbgt_4,772). happensAt(walking(lbgt_4),772).
orientation(lbgt_3,61,773). coords(lbgt_3,146,162,773).  visible(lbgt_3,773). happensAt(inactive(lbgt_3),773). orientation(lbgt_4,141,773). coords(lbgt_4,84,118,773).  visible(lbgt_4,773). happensAt(walking(lbgt_4),773).
orientation(lbgt_3,61,774). coords(lbgt_3,146,162,774).  visible(lbgt_3,774). happensAt(inactive(lbgt_3),774). orientation(lbgt_4,141,774). coords(lbgt_4,84,116,774).  visible(lbgt_4,774). happensAt(walking(lbgt_4),774).
orientation(lbgt_3,61,775). coords(lbgt_3,146,162,775).  visible(lbgt_3,775). happensAt(inactive(lbgt_3),775). orientation(lbgt_4,146,775). coords(lbgt_4,84,114,775).  visible(lbgt_4,775). happensAt(walking(lbgt_4),775).
#show holdsAt/2.