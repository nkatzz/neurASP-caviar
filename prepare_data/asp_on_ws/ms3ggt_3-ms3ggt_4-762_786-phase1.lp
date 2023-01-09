
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

	orientation(ms3ggt_3,148,762). coords(ms3ggt_3,71,77,762).  visible(ms3ggt_3,762). happensAt(walking(ms3ggt_3),762). orientation(ms3ggt_4,129,762). coords(ms3ggt_4,150,22,762).  visible(ms3ggt_4,762). happensAt(walking(ms3ggt_4),762).
orientation(ms3ggt_3,148,763). coords(ms3ggt_3,70,77,763).  visible(ms3ggt_3,763). happensAt(walking(ms3ggt_3),763). orientation(ms3ggt_4,129,763). coords(ms3ggt_4,150,22,763).  visible(ms3ggt_4,763). happensAt(walking(ms3ggt_4),763).
orientation(ms3ggt_3,148,764). coords(ms3ggt_3,68,78,764).  visible(ms3ggt_3,764). happensAt(walking(ms3ggt_3),764). orientation(ms3ggt_4,129,764). coords(ms3ggt_4,150,22,764).  visible(ms3ggt_4,764). happensAt(walking(ms3ggt_4),764).
orientation(ms3ggt_3,148,765). coords(ms3ggt_3,66,79,765).  visible(ms3ggt_3,765). happensAt(walking(ms3ggt_3),765). orientation(ms3ggt_4,129,765). coords(ms3ggt_4,149,23,765).  visible(ms3ggt_4,765). happensAt(walking(ms3ggt_4),765).
orientation(ms3ggt_3,140,766). coords(ms3ggt_3,65,81,766).  visible(ms3ggt_3,766). happensAt(walking(ms3ggt_3),766). orientation(ms3ggt_4,129,766). coords(ms3ggt_4,149,23,766).  visible(ms3ggt_4,766). happensAt(walking(ms3ggt_4),766).
orientation(ms3ggt_3,140,767). coords(ms3ggt_3,65,81,767).  visible(ms3ggt_3,767). happensAt(walking(ms3ggt_3),767). orientation(ms3ggt_4,129,767). coords(ms3ggt_4,149,24,767).  visible(ms3ggt_4,767). happensAt(walking(ms3ggt_4),767).
orientation(ms3ggt_3,140,768). coords(ms3ggt_3,65,82,768).  visible(ms3ggt_3,768). happensAt(walking(ms3ggt_3),768). orientation(ms3ggt_4,129,768). coords(ms3ggt_4,148,24,768).  visible(ms3ggt_4,768). happensAt(walking(ms3ggt_4),768).
orientation(ms3ggt_3,148,769). coords(ms3ggt_3,64,82,769).  visible(ms3ggt_3,769). happensAt(walking(ms3ggt_3),769). orientation(ms3ggt_4,120,769). coords(ms3ggt_4,148,24,769).  visible(ms3ggt_4,769). happensAt(walking(ms3ggt_4),769).
orientation(ms3ggt_3,148,770). coords(ms3ggt_3,64,82,770).  visible(ms3ggt_3,770). happensAt(walking(ms3ggt_3),770). orientation(ms3ggt_4,120,770). coords(ms3ggt_4,147,24,770).  visible(ms3ggt_4,770). happensAt(walking(ms3ggt_4),770).
orientation(ms3ggt_3,148,771). coords(ms3ggt_3,64,83,771).  visible(ms3ggt_3,771). happensAt(walking(ms3ggt_3),771). orientation(ms3ggt_4,120,771). coords(ms3ggt_4,147,24,771).  visible(ms3ggt_4,771). happensAt(walking(ms3ggt_4),771).
orientation(ms3ggt_3,148,772). coords(ms3ggt_3,65,84,772).  visible(ms3ggt_3,772). happensAt(walking(ms3ggt_3),772). orientation(ms3ggt_4,120,772). coords(ms3ggt_4,147,24,772).  visible(ms3ggt_4,772). happensAt(walking(ms3ggt_4),772).
orientation(ms3ggt_3,148,773). coords(ms3ggt_3,63,86,773).  visible(ms3ggt_3,773). happensAt(walking(ms3ggt_3),773). orientation(ms3ggt_4,120,773). coords(ms3ggt_4,147,24,773).  visible(ms3ggt_4,773). happensAt(walking(ms3ggt_4),773).
orientation(ms3ggt_3,148,774). coords(ms3ggt_3,60,87,774).  visible(ms3ggt_3,774). happensAt(walking(ms3ggt_3),774). orientation(ms3ggt_4,120,774). coords(ms3ggt_4,146,24,774).  visible(ms3ggt_4,774). happensAt(walking(ms3ggt_4),774).
orientation(ms3ggt_3,151,775). coords(ms3ggt_3,60,89,775).  visible(ms3ggt_3,775). happensAt(walking(ms3ggt_3),775). orientation(ms3ggt_4,120,775). coords(ms3ggt_4,146,24,775).  visible(ms3ggt_4,775). happensAt(walking(ms3ggt_4),775).
orientation(ms3ggt_3,151,776). coords(ms3ggt_3,60,89,776).  visible(ms3ggt_3,776). happensAt(walking(ms3ggt_3),776). orientation(ms3ggt_4,120,776). coords(ms3ggt_4,146,25,776).  visible(ms3ggt_4,776). happensAt(walking(ms3ggt_4),776).
orientation(ms3ggt_3,151,777). coords(ms3ggt_3,60,90,777).  visible(ms3ggt_3,777). happensAt(walking(ms3ggt_3),777). orientation(ms3ggt_4,120,777). coords(ms3ggt_4,146,25,777).  visible(ms3ggt_4,777). happensAt(walking(ms3ggt_4),777).
orientation(ms3ggt_3,151,778). coords(ms3ggt_3,60,91,778).  visible(ms3ggt_3,778). happensAt(walking(ms3ggt_3),778). orientation(ms3ggt_4,120,778). coords(ms3ggt_4,146,25,778).  visible(ms3ggt_4,778). happensAt(walking(ms3ggt_4),778).
orientation(ms3ggt_3,151,779). coords(ms3ggt_3,59,91,779).  visible(ms3ggt_3,779). happensAt(walking(ms3ggt_3),779). orientation(ms3ggt_4,120,779). coords(ms3ggt_4,146,25,779).  visible(ms3ggt_4,779). happensAt(walking(ms3ggt_4),779).
orientation(ms3ggt_3,151,780). coords(ms3ggt_3,58,92,780).  visible(ms3ggt_3,780). happensAt(walking(ms3ggt_3),780). orientation(ms3ggt_4,120,780). coords(ms3ggt_4,145,26,780).  visible(ms3ggt_4,780). happensAt(walking(ms3ggt_4),780).
orientation(ms3ggt_3,151,781). coords(ms3ggt_3,57,92,781).  visible(ms3ggt_3,781). happensAt(walking(ms3ggt_3),781). orientation(ms3ggt_4,120,781). coords(ms3ggt_4,145,26,781).  visible(ms3ggt_4,781). happensAt(walking(ms3ggt_4),781).
orientation(ms3ggt_3,151,782). coords(ms3ggt_3,57,93,782).  visible(ms3ggt_3,782). happensAt(walking(ms3ggt_3),782). orientation(ms3ggt_4,120,782). coords(ms3ggt_4,145,26,782).  visible(ms3ggt_4,782). happensAt(walking(ms3ggt_4),782).
orientation(ms3ggt_3,151,783). coords(ms3ggt_3,56,94,783).  visible(ms3ggt_3,783). happensAt(walking(ms3ggt_3),783). orientation(ms3ggt_4,120,783). coords(ms3ggt_4,145,26,783).  visible(ms3ggt_4,783). happensAt(walking(ms3ggt_4),783).
orientation(ms3ggt_3,151,784). coords(ms3ggt_3,55,94,784).  visible(ms3ggt_3,784). happensAt(walking(ms3ggt_3),784). orientation(ms3ggt_4,120,784). coords(ms3ggt_4,145,26,784).  visible(ms3ggt_4,784). happensAt(walking(ms3ggt_4),784).
orientation(ms3ggt_3,162,785). coords(ms3ggt_3,54,94,785).  visible(ms3ggt_3,785). happensAt(walking(ms3ggt_3),785). orientation(ms3ggt_4,120,785). coords(ms3ggt_4,144,26,785).  visible(ms3ggt_4,785). happensAt(walking(ms3ggt_4),785).
orientation(ms3ggt_3,162,786). coords(ms3ggt_3,54,97,786).  visible(ms3ggt_3,786). happensAt(walking(ms3ggt_3),786). orientation(ms3ggt_4,120,786). coords(ms3ggt_4,144,26,786).  visible(ms3ggt_4,786). happensAt(walking(ms3ggt_4),786).
#show holdsAt/2.