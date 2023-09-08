
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
orientation(lbgt_3,61,776). coords(lbgt_3,146,162,776).  visible(lbgt_3,776). happensAt(inactive(lbgt_3),776). orientation(lbgt_4,146,776). coords(lbgt_4,84,114,776).  visible(lbgt_4,776). happensAt(walking(lbgt_4),776).
orientation(lbgt_3,61,777). coords(lbgt_3,146,162,777).  visible(lbgt_3,777). happensAt(inactive(lbgt_3),777). orientation(lbgt_4,146,777). coords(lbgt_4,84,114,777).  visible(lbgt_4,777). happensAt(walking(lbgt_4),777).
orientation(lbgt_3,61,778). coords(lbgt_3,146,162,778).  visible(lbgt_3,778). happensAt(inactive(lbgt_3),778). orientation(lbgt_4,146,778). coords(lbgt_4,84,114,778).  visible(lbgt_4,778). happensAt(walking(lbgt_4),778).
orientation(lbgt_3,61,779). coords(lbgt_3,146,162,779).  visible(lbgt_3,779). happensAt(inactive(lbgt_3),779). orientation(lbgt_4,146,779). coords(lbgt_4,84,113,779).  visible(lbgt_4,779). happensAt(walking(lbgt_4),779).
orientation(lbgt_3,61,780). coords(lbgt_3,146,162,780).  visible(lbgt_3,780). happensAt(inactive(lbgt_3),780). orientation(lbgt_4,146,780). coords(lbgt_4,84,113,780).  visible(lbgt_4,780). happensAt(walking(lbgt_4),780).
orientation(lbgt_3,61,781). coords(lbgt_3,146,162,781).  visible(lbgt_3,781). happensAt(inactive(lbgt_3),781). orientation(lbgt_4,146,781). coords(lbgt_4,84,113,781).  visible(lbgt_4,781). happensAt(walking(lbgt_4),781).
orientation(lbgt_3,61,782). coords(lbgt_3,146,162,782).  visible(lbgt_3,782). happensAt(inactive(lbgt_3),782). orientation(lbgt_4,146,782). coords(lbgt_4,84,112,782).  visible(lbgt_4,782). happensAt(walking(lbgt_4),782).
orientation(lbgt_3,61,783). coords(lbgt_3,146,162,783).  visible(lbgt_3,783). happensAt(inactive(lbgt_3),783). orientation(lbgt_4,146,783). coords(lbgt_4,83,111,783).  visible(lbgt_4,783). happensAt(walking(lbgt_4),783).
orientation(lbgt_3,61,784). coords(lbgt_3,146,162,784).  visible(lbgt_3,784). happensAt(inactive(lbgt_3),784). orientation(lbgt_4,149,784). coords(lbgt_4,82,109,784).  visible(lbgt_4,784). happensAt(walking(lbgt_4),784).
orientation(lbgt_3,61,785). coords(lbgt_3,146,162,785).  visible(lbgt_3,785). happensAt(inactive(lbgt_3),785). orientation(lbgt_4,151,785). coords(lbgt_4,82,107,785).  visible(lbgt_4,785). happensAt(walking(lbgt_4),785).
orientation(lbgt_3,61,786). coords(lbgt_3,146,162,786).  visible(lbgt_3,786). happensAt(inactive(lbgt_3),786). orientation(lbgt_4,151,786). coords(lbgt_4,81,107,786).  visible(lbgt_4,786). happensAt(walking(lbgt_4),786).
orientation(lbgt_3,61,787). coords(lbgt_3,146,162,787).  visible(lbgt_3,787). happensAt(inactive(lbgt_3),787). orientation(lbgt_4,151,787). coords(lbgt_4,82,107,787).  visible(lbgt_4,787). happensAt(walking(lbgt_4),787).
#show holdsAt/2.