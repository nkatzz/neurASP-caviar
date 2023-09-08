
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

	orientation(bww1gt_1,173,566). coords(bww1gt_1,99,248,566).  visible(bww1gt_1,566). happensAt(inactive(bww1gt_1),566). orientation(bww1gt_2,169,566). coords(bww1gt_2,151,206,566).  visible(bww1gt_2,566). happensAt(walking(bww1gt_2),566).
orientation(bww1gt_1,173,567). coords(bww1gt_1,99,248,567).  visible(bww1gt_1,567). happensAt(inactive(bww1gt_1),567). orientation(bww1gt_2,169,567). coords(bww1gt_2,151,206,567).  visible(bww1gt_2,567). happensAt(walking(bww1gt_2),567).
orientation(bww1gt_1,173,568). coords(bww1gt_1,99,248,568).  visible(bww1gt_1,568). happensAt(inactive(bww1gt_1),568). orientation(bww1gt_2,169,568). coords(bww1gt_2,150,206,568).  visible(bww1gt_2,568). happensAt(walking(bww1gt_2),568).
orientation(bww1gt_1,173,569). coords(bww1gt_1,99,248,569).  visible(bww1gt_1,569). happensAt(inactive(bww1gt_1),569). orientation(bww1gt_2,169,569). coords(bww1gt_2,150,207,569).  visible(bww1gt_2,569). happensAt(walking(bww1gt_2),569).
orientation(bww1gt_1,173,570). coords(bww1gt_1,99,248,570).  visible(bww1gt_1,570). happensAt(inactive(bww1gt_1),570). orientation(bww1gt_2,169,570). coords(bww1gt_2,149,208,570).  visible(bww1gt_2,570). happensAt(walking(bww1gt_2),570).
orientation(bww1gt_1,173,571). coords(bww1gt_1,99,248,571).  visible(bww1gt_1,571). happensAt(inactive(bww1gt_1),571). orientation(bww1gt_2,169,571). coords(bww1gt_2,148,209,571).  visible(bww1gt_2,571). happensAt(walking(bww1gt_2),571).
orientation(bww1gt_1,173,572). coords(bww1gt_1,99,249,572).  visible(bww1gt_1,572). happensAt(inactive(bww1gt_1),572). orientation(bww1gt_2,169,572). coords(bww1gt_2,147,209,572).  visible(bww1gt_2,572). happensAt(walking(bww1gt_2),572).
orientation(bww1gt_1,173,573). coords(bww1gt_1,99,249,573).  visible(bww1gt_1,573). happensAt(inactive(bww1gt_1),573). orientation(bww1gt_2,169,573). coords(bww1gt_2,145,209,573).  visible(bww1gt_2,573). happensAt(walking(bww1gt_2),573).
orientation(bww1gt_1,173,574). coords(bww1gt_1,99,249,574).  visible(bww1gt_1,574). happensAt(inactive(bww1gt_1),574). orientation(bww1gt_2,169,574). coords(bww1gt_2,144,209,574).  visible(bww1gt_2,574). happensAt(walking(bww1gt_2),574).
orientation(bww1gt_1,173,575). coords(bww1gt_1,99,249,575).  visible(bww1gt_1,575). happensAt(inactive(bww1gt_1),575). orientation(bww1gt_2,169,575). coords(bww1gt_2,143,209,575).  visible(bww1gt_2,575). happensAt(walking(bww1gt_2),575).
orientation(bww1gt_1,173,576). coords(bww1gt_1,99,249,576).  visible(bww1gt_1,576). happensAt(inactive(bww1gt_1),576). orientation(bww1gt_2,169,576). coords(bww1gt_2,141,210,576).  visible(bww1gt_2,576). happensAt(walking(bww1gt_2),576).
orientation(bww1gt_1,173,577). coords(bww1gt_1,99,249,577).  visible(bww1gt_1,577). happensAt(inactive(bww1gt_1),577). orientation(bww1gt_2,169,577). coords(bww1gt_2,141,210,577).  visible(bww1gt_2,577). happensAt(walking(bww1gt_2),577).
orientation(bww1gt_1,173,578). coords(bww1gt_1,99,249,578).  visible(bww1gt_1,578). happensAt(inactive(bww1gt_1),578). orientation(bww1gt_2,176,578). coords(bww1gt_2,139,210,578).  visible(bww1gt_2,578). happensAt(walking(bww1gt_2),578).
orientation(bww1gt_1,173,579). coords(bww1gt_1,100,249,579).  visible(bww1gt_1,579). happensAt(inactive(bww1gt_1),579). orientation(bww1gt_2,178,579). coords(bww1gt_2,138,210,579).  visible(bww1gt_2,579). happensAt(walking(bww1gt_2),579).
orientation(bww1gt_1,173,580). coords(bww1gt_1,100,249,580).  visible(bww1gt_1,580). happensAt(inactive(bww1gt_1),580). orientation(bww1gt_2,1,580). coords(bww1gt_2,138,210,580).  visible(bww1gt_2,580). happensAt(walking(bww1gt_2),580).
orientation(bww1gt_1,173,581). coords(bww1gt_1,100,249,581).  visible(bww1gt_1,581). happensAt(inactive(bww1gt_1),581). orientation(bww1gt_2,179,581). coords(bww1gt_2,137,210,581).  visible(bww1gt_2,581). happensAt(walking(bww1gt_2),581).
orientation(bww1gt_1,173,582). coords(bww1gt_1,100,249,582).  visible(bww1gt_1,582). happensAt(inactive(bww1gt_1),582). orientation(bww1gt_2,1,582). coords(bww1gt_2,137,210,582).  visible(bww1gt_2,582). happensAt(walking(bww1gt_2),582).
orientation(bww1gt_1,173,583). coords(bww1gt_1,100,249,583).  visible(bww1gt_1,583). happensAt(inactive(bww1gt_1),583). orientation(bww1gt_2,1,583). coords(bww1gt_2,136,210,583).  visible(bww1gt_2,583). happensAt(walking(bww1gt_2),583).
orientation(bww1gt_1,173,584). coords(bww1gt_1,100,249,584).  visible(bww1gt_1,584). happensAt(inactive(bww1gt_1),584). orientation(bww1gt_2,2,584). coords(bww1gt_2,136,210,584).  visible(bww1gt_2,584). happensAt(walking(bww1gt_2),584).
orientation(bww1gt_1,173,585). coords(bww1gt_1,100,249,585).  visible(bww1gt_1,585). happensAt(inactive(bww1gt_1),585). orientation(bww1gt_2,0,585). coords(bww1gt_2,136,211,585).  visible(bww1gt_2,585). happensAt(walking(bww1gt_2),585).
orientation(bww1gt_1,173,586). coords(bww1gt_1,100,249,586).  visible(bww1gt_1,586). happensAt(inactive(bww1gt_1),586). orientation(bww1gt_2,176,586). coords(bww1gt_2,136,211,586).  visible(bww1gt_2,586). happensAt(walking(bww1gt_2),586).
orientation(bww1gt_1,173,587). coords(bww1gt_1,100,249,587).  visible(bww1gt_1,587). happensAt(inactive(bww1gt_1),587). orientation(bww1gt_2,0,587). coords(bww1gt_2,136,211,587).  visible(bww1gt_2,587). happensAt(walking(bww1gt_2),587).
orientation(bww1gt_1,173,588). coords(bww1gt_1,100,249,588).  visible(bww1gt_1,588). happensAt(inactive(bww1gt_1),588). orientation(bww1gt_2,179,588). coords(bww1gt_2,135,211,588).  visible(bww1gt_2,588). happensAt(walking(bww1gt_2),588).
orientation(bww1gt_1,173,589). coords(bww1gt_1,100,249,589).  visible(bww1gt_1,589). happensAt(inactive(bww1gt_1),589). orientation(bww1gt_2,176,589). coords(bww1gt_2,135,211,589).  visible(bww1gt_2,589). happensAt(walking(bww1gt_2),589).
orientation(bww1gt_1,173,590). coords(bww1gt_1,100,249,590).  visible(bww1gt_1,590). happensAt(inactive(bww1gt_1),590). orientation(bww1gt_2,176,590). coords(bww1gt_2,135,211,590).  visible(bww1gt_2,590). happensAt(walking(bww1gt_2),590).
#show holdsAt/2.