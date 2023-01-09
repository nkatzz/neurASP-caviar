
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

	orientation(mwt2gt_1,156,493). coords(mwt2gt_1,59,148,493).  visible(mwt2gt_1,493). happensAt(walking(mwt2gt_1),493). orientation(mwt2gt_2,146,493). coords(mwt2gt_2,53,158,493).  visible(mwt2gt_2,493). happensAt(walking(mwt2gt_2),493).
orientation(mwt2gt_1,156,494). coords(mwt2gt_1,58,148,494).  visible(mwt2gt_1,494). happensAt(walking(mwt2gt_1),494). orientation(mwt2gt_2,144,494). coords(mwt2gt_2,53,160,494).  visible(mwt2gt_2,494). happensAt(walking(mwt2gt_2),494).
orientation(mwt2gt_1,156,495). coords(mwt2gt_1,57,148,495).  visible(mwt2gt_1,495). happensAt(walking(mwt2gt_1),495). orientation(mwt2gt_2,149,495). coords(mwt2gt_2,53,160,495).  visible(mwt2gt_2,495). happensAt(walking(mwt2gt_2),495).
orientation(mwt2gt_1,156,496). coords(mwt2gt_1,56,149,496).  visible(mwt2gt_1,496). happensAt(walking(mwt2gt_1),496). orientation(mwt2gt_2,151,496). coords(mwt2gt_2,53,161,496).  visible(mwt2gt_2,496). happensAt(walking(mwt2gt_2),496).
orientation(mwt2gt_1,156,497). coords(mwt2gt_1,55,149,497).  visible(mwt2gt_1,497). happensAt(walking(mwt2gt_1),497). orientation(mwt2gt_2,156,497). coords(mwt2gt_2,53,161,497).  visible(mwt2gt_2,497). happensAt(walking(mwt2gt_2),497).
orientation(mwt2gt_1,156,498). coords(mwt2gt_1,55,149,498).  visible(mwt2gt_1,498). happensAt(walking(mwt2gt_1),498). orientation(mwt2gt_2,153,498). coords(mwt2gt_2,53,162,498).  visible(mwt2gt_2,498). happensAt(walking(mwt2gt_2),498).
orientation(mwt2gt_1,156,499). coords(mwt2gt_1,55,149,499).  visible(mwt2gt_1,499). happensAt(walking(mwt2gt_1),499). orientation(mwt2gt_2,151,499). coords(mwt2gt_2,52,162,499).  visible(mwt2gt_2,499). happensAt(walking(mwt2gt_2),499).
orientation(mwt2gt_1,156,500). coords(mwt2gt_1,55,150,500).  visible(mwt2gt_1,500). happensAt(walking(mwt2gt_1),500). orientation(mwt2gt_2,154,500). coords(mwt2gt_2,52,162,500).  visible(mwt2gt_2,500). happensAt(walking(mwt2gt_2),500).
orientation(mwt2gt_1,156,501). coords(mwt2gt_1,55,150,501).  visible(mwt2gt_1,501). happensAt(walking(mwt2gt_1),501). orientation(mwt2gt_2,153,501). coords(mwt2gt_2,51,163,501).  visible(mwt2gt_2,501). happensAt(walking(mwt2gt_2),501).
orientation(mwt2gt_1,156,502). coords(mwt2gt_1,54,151,502).  visible(mwt2gt_1,502). happensAt(walking(mwt2gt_1),502). orientation(mwt2gt_2,153,502). coords(mwt2gt_2,50,163,502).  visible(mwt2gt_2,502). happensAt(walking(mwt2gt_2),502).
orientation(mwt2gt_1,156,503). coords(mwt2gt_1,54,152,503).  visible(mwt2gt_1,503). happensAt(walking(mwt2gt_1),503). orientation(mwt2gt_2,153,503). coords(mwt2gt_2,49,163,503).  visible(mwt2gt_2,503). happensAt(walking(mwt2gt_2),503).
orientation(mwt2gt_1,156,504). coords(mwt2gt_1,54,152,504).  visible(mwt2gt_1,504). happensAt(walking(mwt2gt_1),504). orientation(mwt2gt_2,152,504). coords(mwt2gt_2,48,163,504).  visible(mwt2gt_2,504). happensAt(walking(mwt2gt_2),504).
orientation(mwt2gt_1,156,505). coords(mwt2gt_1,54,153,505).  visible(mwt2gt_1,505). happensAt(walking(mwt2gt_1),505). orientation(mwt2gt_2,151,505). coords(mwt2gt_2,48,163,505).  visible(mwt2gt_2,505). happensAt(walking(mwt2gt_2),505).
orientation(mwt2gt_1,156,506). coords(mwt2gt_1,53,153,506).  visible(mwt2gt_1,506). happensAt(walking(mwt2gt_1),506). orientation(mwt2gt_2,153,506). coords(mwt2gt_2,48,164,506).  visible(mwt2gt_2,506). happensAt(walking(mwt2gt_2),506).
orientation(mwt2gt_1,156,507). coords(mwt2gt_1,53,153,507).  visible(mwt2gt_1,507). happensAt(walking(mwt2gt_1),507). orientation(mwt2gt_2,149,507). coords(mwt2gt_2,48,164,507).  visible(mwt2gt_2,507). happensAt(walking(mwt2gt_2),507).
orientation(mwt2gt_1,156,508). coords(mwt2gt_1,51,153,508).  visible(mwt2gt_1,508). happensAt(walking(mwt2gt_1),508). orientation(mwt2gt_2,156,508). coords(mwt2gt_2,48,164,508).  visible(mwt2gt_2,508). happensAt(walking(mwt2gt_2),508).
orientation(mwt2gt_1,156,509). coords(mwt2gt_1,50,154,509).  visible(mwt2gt_1,509). happensAt(walking(mwt2gt_1),509). orientation(mwt2gt_2,152,509). coords(mwt2gt_2,48,164,509).  visible(mwt2gt_2,509). happensAt(walking(mwt2gt_2),509).
orientation(mwt2gt_1,156,510). coords(mwt2gt_1,50,154,510).  visible(mwt2gt_1,510). happensAt(walking(mwt2gt_1),510). orientation(mwt2gt_2,152,510). coords(mwt2gt_2,46,165,510).  visible(mwt2gt_2,510). happensAt(walking(mwt2gt_2),510).
orientation(mwt2gt_1,153,511). coords(mwt2gt_1,50,154,511).  visible(mwt2gt_1,511). happensAt(walking(mwt2gt_1),511). orientation(mwt2gt_2,153,511). coords(mwt2gt_2,46,165,511).  visible(mwt2gt_2,511). happensAt(walking(mwt2gt_2),511).
orientation(mwt2gt_1,153,512). coords(mwt2gt_1,50,155,512).  visible(mwt2gt_1,512). happensAt(walking(mwt2gt_1),512). orientation(mwt2gt_2,152,512). coords(mwt2gt_2,46,165,512).  visible(mwt2gt_2,512). happensAt(walking(mwt2gt_2),512).
orientation(mwt2gt_1,153,513). coords(mwt2gt_1,50,155,513).  visible(mwt2gt_1,513). happensAt(walking(mwt2gt_1),513). orientation(mwt2gt_2,152,513). coords(mwt2gt_2,46,166,513).  visible(mwt2gt_2,513). happensAt(walking(mwt2gt_2),513).
orientation(mwt2gt_1,153,514). coords(mwt2gt_1,50,155,514).  visible(mwt2gt_1,514). happensAt(walking(mwt2gt_1),514). orientation(mwt2gt_2,152,514). coords(mwt2gt_2,45,166,514).  visible(mwt2gt_2,514). happensAt(walking(mwt2gt_2),514).
orientation(mwt2gt_1,153,515). coords(mwt2gt_1,50,155,515).  visible(mwt2gt_1,515). happensAt(walking(mwt2gt_1),515). orientation(mwt2gt_2,152,515). coords(mwt2gt_2,45,166,515).  visible(mwt2gt_2,515). happensAt(walking(mwt2gt_2),515).
orientation(mwt2gt_1,153,516). coords(mwt2gt_1,50,156,516).  visible(mwt2gt_1,516). happensAt(walking(mwt2gt_1),516). orientation(mwt2gt_2,152,516). coords(mwt2gt_2,45,166,516).  visible(mwt2gt_2,516). happensAt(walking(mwt2gt_2),516).
orientation(mwt2gt_1,153,517). coords(mwt2gt_1,50,155,517).  visible(mwt2gt_1,517). happensAt(walking(mwt2gt_1),517). orientation(mwt2gt_2,152,517). coords(mwt2gt_2,44,166,517).  visible(mwt2gt_2,517). happensAt(walking(mwt2gt_2),517).
#show holdsAt/2.