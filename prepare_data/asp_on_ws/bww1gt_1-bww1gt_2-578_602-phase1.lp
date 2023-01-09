
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
orientation(bww1gt_1,173,591). coords(bww1gt_1,100,249,591).  visible(bww1gt_1,591). happensAt(inactive(bww1gt_1),591). orientation(bww1gt_2,173,591). coords(bww1gt_2,135,211,591).  visible(bww1gt_2,591). happensAt(walking(bww1gt_2),591).
orientation(bww1gt_1,173,592). coords(bww1gt_1,100,249,592).  visible(bww1gt_1,592). happensAt(walking(bww1gt_1),592). orientation(bww1gt_2,177,592). coords(bww1gt_2,135,211,592).  visible(bww1gt_2,592). happensAt(walking(bww1gt_2),592).
orientation(bww1gt_1,173,593). coords(bww1gt_1,100,249,593).  visible(bww1gt_1,593). happensAt(walking(bww1gt_1),593). orientation(bww1gt_2,177,593). coords(bww1gt_2,134,211,593).  visible(bww1gt_2,593). happensAt(walking(bww1gt_2),593).
orientation(bww1gt_1,173,594). coords(bww1gt_1,100,249,594).  visible(bww1gt_1,594). happensAt(walking(bww1gt_1),594). orientation(bww1gt_2,175,594). coords(bww1gt_2,134,211,594).  visible(bww1gt_2,594). happensAt(walking(bww1gt_2),594).
orientation(bww1gt_1,173,595). coords(bww1gt_1,100,249,595).  visible(bww1gt_1,595). happensAt(walking(bww1gt_1),595). orientation(bww1gt_2,177,595). coords(bww1gt_2,134,211,595).  visible(bww1gt_2,595). happensAt(walking(bww1gt_2),595).
orientation(bww1gt_1,173,596). coords(bww1gt_1,100,249,596).  visible(bww1gt_1,596). happensAt(walking(bww1gt_1),596). orientation(bww1gt_2,1,596). coords(bww1gt_2,134,211,596).  visible(bww1gt_2,596). happensAt(walking(bww1gt_2),596).
orientation(bww1gt_1,173,597). coords(bww1gt_1,100,249,597).  visible(bww1gt_1,597). happensAt(walking(bww1gt_1),597). orientation(bww1gt_2,0,597). coords(bww1gt_2,134,211,597).  visible(bww1gt_2,597). happensAt(walking(bww1gt_2),597).
orientation(bww1gt_1,173,598). coords(bww1gt_1,100,249,598).  visible(bww1gt_1,598). happensAt(walking(bww1gt_1),598). orientation(bww1gt_2,178,598). coords(bww1gt_2,134,211,598).  visible(bww1gt_2,598). happensAt(walking(bww1gt_2),598).
orientation(bww1gt_1,173,599). coords(bww1gt_1,100,249,599).  visible(bww1gt_1,599). happensAt(walking(bww1gt_1),599). orientation(bww1gt_2,1,599). coords(bww1gt_2,134,211,599).  visible(bww1gt_2,599). happensAt(active(bww1gt_2),599).
orientation(bww1gt_1,173,600). coords(bww1gt_1,101,250,600).  visible(bww1gt_1,600). happensAt(walking(bww1gt_1),600). orientation(bww1gt_2,178,600). coords(bww1gt_2,134,211,600).  visible(bww1gt_2,600). happensAt(active(bww1gt_2),600).
orientation(bww1gt_1,173,601). coords(bww1gt_1,101,250,601).  visible(bww1gt_1,601). happensAt(walking(bww1gt_1),601). orientation(bww1gt_2,1,601). coords(bww1gt_2,134,211,601).  visible(bww1gt_2,601). happensAt(active(bww1gt_2),601).
orientation(bww1gt_1,173,602). coords(bww1gt_1,101,250,602).  visible(bww1gt_1,602). happensAt(walking(bww1gt_1),602). orientation(bww1gt_2,0,602). coords(bww1gt_2,134,211,602).  visible(bww1gt_2,602). happensAt(active(bww1gt_2),602).
#show holdsAt/2.