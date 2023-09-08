
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

	orientation(fra2gt_3,144,490). coords(fra2gt_3,39,145,490).  visible(fra2gt_3,490). happensAt(walking(fra2gt_3),490). orientation(fra2gt_4,150,490). coords(fra2gt_4,64,81,490).  visible(fra2gt_4,490). happensAt(walking(fra2gt_4),490).
orientation(fra2gt_3,144,491). coords(fra2gt_3,39,145,491).  visible(fra2gt_3,491). happensAt(walking(fra2gt_3),491). orientation(fra2gt_4,146,491). coords(fra2gt_4,63,82,491).  visible(fra2gt_4,491). happensAt(walking(fra2gt_4),491).
orientation(fra2gt_3,144,492). coords(fra2gt_3,39,142,492).  visible(fra2gt_3,492). happensAt(walking(fra2gt_3),492). orientation(fra2gt_4,146,492). coords(fra2gt_4,63,82,492).  visible(fra2gt_4,492). happensAt(walking(fra2gt_4),492).
orientation(fra2gt_3,144,493). coords(fra2gt_3,39,141,493).  visible(fra2gt_3,493). happensAt(walking(fra2gt_3),493). orientation(fra2gt_4,146,493). coords(fra2gt_4,63,82,493).  visible(fra2gt_4,493). happensAt(walking(fra2gt_4),493).
orientation(fra2gt_3,144,494). coords(fra2gt_3,39,141,494).  visible(fra2gt_3,494). happensAt(walking(fra2gt_3),494). orientation(fra2gt_4,146,494). coords(fra2gt_4,63,82,494).  visible(fra2gt_4,494). happensAt(walking(fra2gt_4),494).
orientation(fra2gt_3,144,495). coords(fra2gt_3,39,139,495).  visible(fra2gt_3,495). happensAt(walking(fra2gt_3),495). orientation(fra2gt_4,146,495). coords(fra2gt_4,63,82,495).  visible(fra2gt_4,495). happensAt(walking(fra2gt_4),495).
orientation(fra2gt_3,144,496). coords(fra2gt_3,39,138,496).  visible(fra2gt_3,496). happensAt(walking(fra2gt_3),496). orientation(fra2gt_4,146,496). coords(fra2gt_4,63,82,496).  visible(fra2gt_4,496). happensAt(walking(fra2gt_4),496).
orientation(fra2gt_3,144,497). coords(fra2gt_3,39,137,497).  visible(fra2gt_3,497). happensAt(walking(fra2gt_3),497). orientation(fra2gt_4,146,497). coords(fra2gt_4,63,82,497).  visible(fra2gt_4,497). happensAt(walking(fra2gt_4),497).
orientation(fra2gt_3,144,498). coords(fra2gt_3,39,136,498).  visible(fra2gt_3,498). happensAt(walking(fra2gt_3),498). orientation(fra2gt_4,146,498). coords(fra2gt_4,63,82,498).  visible(fra2gt_4,498). happensAt(walking(fra2gt_4),498).
orientation(fra2gt_3,144,499). coords(fra2gt_3,40,136,499).  visible(fra2gt_3,499). happensAt(walking(fra2gt_3),499). orientation(fra2gt_4,146,499). coords(fra2gt_4,63,82,499).  visible(fra2gt_4,499). happensAt(walking(fra2gt_4),499).
orientation(fra2gt_3,135,500). coords(fra2gt_3,41,135,500).  visible(fra2gt_3,500). happensAt(walking(fra2gt_3),500). orientation(fra2gt_4,146,500). coords(fra2gt_4,63,82,500).  visible(fra2gt_4,500). happensAt(walking(fra2gt_4),500).
orientation(fra2gt_3,144,501). coords(fra2gt_3,41,134,501).  visible(fra2gt_3,501). happensAt(walking(fra2gt_3),501). orientation(fra2gt_4,146,501). coords(fra2gt_4,63,82,501).  visible(fra2gt_4,501). happensAt(walking(fra2gt_4),501).
orientation(fra2gt_3,153,502). coords(fra2gt_3,41,132,502).  visible(fra2gt_3,502). happensAt(walking(fra2gt_3),502). orientation(fra2gt_4,146,502). coords(fra2gt_4,63,82,502).  visible(fra2gt_4,502). happensAt(walking(fra2gt_4),502).
orientation(fra2gt_3,153,503). coords(fra2gt_3,42,132,503).  visible(fra2gt_3,503). happensAt(walking(fra2gt_3),503). orientation(fra2gt_4,146,503). coords(fra2gt_4,63,82,503).  visible(fra2gt_4,503). happensAt(walking(fra2gt_4),503).
orientation(fra2gt_3,153,504). coords(fra2gt_3,42,132,504).  visible(fra2gt_3,504). happensAt(walking(fra2gt_3),504). orientation(fra2gt_4,146,504). coords(fra2gt_4,63,82,504).  visible(fra2gt_4,504). happensAt(walking(fra2gt_4),504).
orientation(fra2gt_3,153,505). coords(fra2gt_3,42,131,505).  visible(fra2gt_3,505). happensAt(walking(fra2gt_3),505). orientation(fra2gt_4,146,505). coords(fra2gt_4,63,82,505).  visible(fra2gt_4,505). happensAt(walking(fra2gt_4),505).
orientation(fra2gt_3,144,506). coords(fra2gt_3,42,131,506).  visible(fra2gt_3,506). happensAt(walking(fra2gt_3),506). orientation(fra2gt_4,146,506). coords(fra2gt_4,63,82,506).  visible(fra2gt_4,506). happensAt(walking(fra2gt_4),506).
orientation(fra2gt_3,142,507). coords(fra2gt_3,42,131,507).  visible(fra2gt_3,507). happensAt(walking(fra2gt_3),507). orientation(fra2gt_4,146,507). coords(fra2gt_4,63,82,507).  visible(fra2gt_4,507). happensAt(walking(fra2gt_4),507).
orientation(fra2gt_3,144,508). coords(fra2gt_3,42,129,508).  visible(fra2gt_3,508). happensAt(walking(fra2gt_3),508). orientation(fra2gt_4,146,508). coords(fra2gt_4,63,82,508).  visible(fra2gt_4,508). happensAt(walking(fra2gt_4),508).
orientation(fra2gt_3,143,509). coords(fra2gt_3,42,129,509).  visible(fra2gt_3,509). happensAt(walking(fra2gt_3),509). orientation(fra2gt_4,146,509). coords(fra2gt_4,63,82,509).  visible(fra2gt_4,509). happensAt(walking(fra2gt_4),509).
orientation(fra2gt_3,141,510). coords(fra2gt_3,42,129,510).  visible(fra2gt_3,510). happensAt(walking(fra2gt_3),510). orientation(fra2gt_4,146,510). coords(fra2gt_4,63,82,510).  visible(fra2gt_4,510). happensAt(walking(fra2gt_4),510).
orientation(fra2gt_3,139,511). coords(fra2gt_3,42,128,511).  visible(fra2gt_3,511). happensAt(walking(fra2gt_3),511). orientation(fra2gt_4,146,511). coords(fra2gt_4,63,82,511).  visible(fra2gt_4,511). happensAt(walking(fra2gt_4),511).
orientation(fra2gt_3,143,512). coords(fra2gt_3,42,127,512).  visible(fra2gt_3,512). happensAt(walking(fra2gt_3),512). orientation(fra2gt_4,146,512). coords(fra2gt_4,63,82,512).  visible(fra2gt_4,512). happensAt(walking(fra2gt_4),512).
orientation(fra2gt_3,139,513). coords(fra2gt_3,43,127,513).  visible(fra2gt_3,513). happensAt(walking(fra2gt_3),513). orientation(fra2gt_4,146,513). coords(fra2gt_4,63,82,513).  visible(fra2gt_4,513). happensAt(walking(fra2gt_4),513).
orientation(fra2gt_3,139,514). coords(fra2gt_3,44,126,514).  visible(fra2gt_3,514). happensAt(walking(fra2gt_3),514). orientation(fra2gt_4,146,514). coords(fra2gt_4,63,82,514).  visible(fra2gt_4,514). happensAt(walking(fra2gt_4),514).
#show holdsAt/2.