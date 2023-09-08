
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

	orientation(br1gt_2,12,490). coords(br1gt_2,86,209,490).  visible(br1gt_2,490). happensAt(inactive(br1gt_2),490). orientation(br1gt_3,160,490). coords(br1gt_3,99,140,490).  visible(br1gt_3,490). happensAt(walking(br1gt_3),490).
orientation(br1gt_2,12,491). coords(br1gt_2,86,209,491).  visible(br1gt_2,491). happensAt(inactive(br1gt_2),491). orientation(br1gt_3,160,491). coords(br1gt_3,101,139,491).  visible(br1gt_3,491). happensAt(walking(br1gt_3),491).
orientation(br1gt_2,12,492). coords(br1gt_2,86,209,492).  visible(br1gt_2,492). happensAt(inactive(br1gt_2),492). orientation(br1gt_3,160,492). coords(br1gt_3,101,139,492).  visible(br1gt_3,492). happensAt(walking(br1gt_3),492).
orientation(br1gt_2,12,493). coords(br1gt_2,86,209,493).  visible(br1gt_2,493). happensAt(inactive(br1gt_2),493). orientation(br1gt_3,160,493). coords(br1gt_3,102,139,493).  visible(br1gt_3,493). happensAt(walking(br1gt_3),493).
orientation(br1gt_2,12,494). coords(br1gt_2,86,209,494).  visible(br1gt_2,494). happensAt(inactive(br1gt_2),494). orientation(br1gt_3,160,494). coords(br1gt_3,102,137,494).  visible(br1gt_3,494). happensAt(walking(br1gt_3),494).
orientation(br1gt_2,12,495). coords(br1gt_2,86,209,495).  visible(br1gt_2,495). happensAt(inactive(br1gt_2),495). orientation(br1gt_3,160,495). coords(br1gt_3,103,137,495).  visible(br1gt_3,495). happensAt(walking(br1gt_3),495).
orientation(br1gt_2,12,496). coords(br1gt_2,86,209,496).  visible(br1gt_2,496). happensAt(inactive(br1gt_2),496). orientation(br1gt_3,160,496). coords(br1gt_3,103,137,496).  visible(br1gt_3,496). happensAt(walking(br1gt_3),496).
orientation(br1gt_2,12,497). coords(br1gt_2,86,209,497).  visible(br1gt_2,497). happensAt(inactive(br1gt_2),497). orientation(br1gt_3,160,497). coords(br1gt_3,103,136,497).  visible(br1gt_3,497). happensAt(walking(br1gt_3),497).
orientation(br1gt_2,12,498). coords(br1gt_2,86,209,498).  visible(br1gt_2,498). happensAt(inactive(br1gt_2),498). orientation(br1gt_3,157,498). coords(br1gt_3,104,136,498).  visible(br1gt_3,498). happensAt(walking(br1gt_3),498).
orientation(br1gt_2,12,499). coords(br1gt_2,86,209,499).  visible(br1gt_2,499). happensAt(inactive(br1gt_2),499). orientation(br1gt_3,157,499). coords(br1gt_3,106,135,499).  visible(br1gt_3,499). happensAt(walking(br1gt_3),499).
orientation(br1gt_2,12,500). coords(br1gt_2,86,209,500).  visible(br1gt_2,500). happensAt(inactive(br1gt_2),500). orientation(br1gt_3,157,500). coords(br1gt_3,109,134,500).  visible(br1gt_3,500). happensAt(walking(br1gt_3),500).
orientation(br1gt_2,12,501). coords(br1gt_2,86,209,501).  visible(br1gt_2,501). happensAt(inactive(br1gt_2),501). orientation(br1gt_3,157,501). coords(br1gt_3,110,133,501).  visible(br1gt_3,501). happensAt(walking(br1gt_3),501).
orientation(br1gt_2,12,502). coords(br1gt_2,86,209,502).  visible(br1gt_2,502). happensAt(inactive(br1gt_2),502). orientation(br1gt_3,157,502). coords(br1gt_3,112,133,502).  visible(br1gt_3,502). happensAt(walking(br1gt_3),502).
orientation(br1gt_2,12,503). coords(br1gt_2,86,209,503).  visible(br1gt_2,503). happensAt(inactive(br1gt_2),503). orientation(br1gt_3,157,503). coords(br1gt_3,113,133,503).  visible(br1gt_3,503). happensAt(walking(br1gt_3),503).
orientation(br1gt_2,12,504). coords(br1gt_2,86,209,504).  visible(br1gt_2,504). happensAt(inactive(br1gt_2),504). orientation(br1gt_3,157,504). coords(br1gt_3,115,133,504).  visible(br1gt_3,504). happensAt(walking(br1gt_3),504).
orientation(br1gt_2,12,505). coords(br1gt_2,86,209,505).  visible(br1gt_2,505). happensAt(inactive(br1gt_2),505). orientation(br1gt_3,157,505). coords(br1gt_3,115,133,505).  visible(br1gt_3,505). happensAt(walking(br1gt_3),505).
orientation(br1gt_2,12,506). coords(br1gt_2,86,209,506).  visible(br1gt_2,506). happensAt(inactive(br1gt_2),506). orientation(br1gt_3,157,506). coords(br1gt_3,116,133,506).  visible(br1gt_3,506). happensAt(walking(br1gt_3),506).
orientation(br1gt_2,12,507). coords(br1gt_2,86,209,507).  visible(br1gt_2,507). happensAt(inactive(br1gt_2),507). orientation(br1gt_3,157,507). coords(br1gt_3,116,132,507).  visible(br1gt_3,507). happensAt(walking(br1gt_3),507).
orientation(br1gt_2,12,508). coords(br1gt_2,86,209,508).  visible(br1gt_2,508). happensAt(inactive(br1gt_2),508). orientation(br1gt_3,157,508). coords(br1gt_3,117,132,508).  visible(br1gt_3,508). happensAt(walking(br1gt_3),508).
orientation(br1gt_2,12,509). coords(br1gt_2,86,209,509).  visible(br1gt_2,509). happensAt(inactive(br1gt_2),509). orientation(br1gt_3,157,509). coords(br1gt_3,117,132,509).  visible(br1gt_3,509). happensAt(walking(br1gt_3),509).
orientation(br1gt_2,12,510). coords(br1gt_2,86,209,510).  visible(br1gt_2,510). happensAt(inactive(br1gt_2),510). orientation(br1gt_3,157,510). coords(br1gt_3,118,132,510).  visible(br1gt_3,510). happensAt(walking(br1gt_3),510).
orientation(br1gt_2,12,511). coords(br1gt_2,86,209,511).  visible(br1gt_2,511). happensAt(inactive(br1gt_2),511). orientation(br1gt_3,155,511). coords(br1gt_3,118,132,511).  visible(br1gt_3,511). happensAt(walking(br1gt_3),511).
orientation(br1gt_2,12,512). coords(br1gt_2,86,209,512).  visible(br1gt_2,512). happensAt(inactive(br1gt_2),512). orientation(br1gt_3,157,512). coords(br1gt_3,122,131,512).  visible(br1gt_3,512). happensAt(walking(br1gt_3),512).
orientation(br1gt_2,12,513). coords(br1gt_2,86,209,513).  visible(br1gt_2,513). happensAt(inactive(br1gt_2),513). orientation(br1gt_3,157,513). coords(br1gt_3,124,131,513).  visible(br1gt_3,513). happensAt(walking(br1gt_3),513).
orientation(br1gt_2,12,514). coords(br1gt_2,86,209,514).  visible(br1gt_2,514). happensAt(inactive(br1gt_2),514). orientation(br1gt_3,157,514). coords(br1gt_3,127,130,514).  visible(br1gt_3,514). happensAt(walking(br1gt_3),514).
#show holdsAt/2.