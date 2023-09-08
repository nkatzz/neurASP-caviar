
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

	orientation(lb1gt_1,151,288). coords(lb1gt_1,155,158,288).  visible(lb1gt_1,288). happensAt(active(lb1gt_1),288). orientation(lb1gt_2,146,288). coords(lb1gt_2,186,152,288).  visible(lb1gt_2,288). happensAt(walking(lb1gt_2),288).
orientation(lb1gt_1,151,289). coords(lb1gt_1,156,158,289).  visible(lb1gt_1,289). happensAt(active(lb1gt_1),289). orientation(lb1gt_2,146,289). coords(lb1gt_2,187,152,289).  visible(lb1gt_2,289). happensAt(walking(lb1gt_2),289).
orientation(lb1gt_1,151,290). coords(lb1gt_1,156,158,290).  visible(lb1gt_1,290). happensAt(walking(lb1gt_1),290). orientation(lb1gt_2,146,290). coords(lb1gt_2,189,152,290).  visible(lb1gt_2,290). happensAt(walking(lb1gt_2),290).
orientation(lb1gt_1,151,291). coords(lb1gt_1,156,158,291).  visible(lb1gt_1,291). happensAt(walking(lb1gt_1),291). orientation(lb1gt_2,146,291). coords(lb1gt_2,190,152,291).  visible(lb1gt_2,291). happensAt(walking(lb1gt_2),291).
orientation(lb1gt_1,151,292). coords(lb1gt_1,156,158,292).  visible(lb1gt_1,292). happensAt(walking(lb1gt_1),292). orientation(lb1gt_2,146,292). coords(lb1gt_2,191,153,292).  visible(lb1gt_2,292). happensAt(walking(lb1gt_2),292).
orientation(lb1gt_1,151,293). coords(lb1gt_1,156,159,293).  visible(lb1gt_1,293). happensAt(walking(lb1gt_1),293). orientation(lb1gt_2,146,293). coords(lb1gt_2,192,153,293).  visible(lb1gt_2,293). happensAt(walking(lb1gt_2),293).
orientation(lb1gt_1,151,294). coords(lb1gt_1,156,160,294).  visible(lb1gt_1,294). happensAt(walking(lb1gt_1),294). orientation(lb1gt_2,146,294). coords(lb1gt_2,192,153,294).  visible(lb1gt_2,294). happensAt(walking(lb1gt_2),294).
orientation(lb1gt_1,151,295). coords(lb1gt_1,156,160,295).  visible(lb1gt_1,295). happensAt(walking(lb1gt_1),295). orientation(lb1gt_2,146,295). coords(lb1gt_2,193,153,295).  visible(lb1gt_2,295). happensAt(active(lb1gt_2),295).
orientation(lb1gt_1,151,296). coords(lb1gt_1,157,160,296).  visible(lb1gt_1,296). happensAt(walking(lb1gt_1),296). orientation(lb1gt_2,146,296). coords(lb1gt_2,193,153,296).  visible(lb1gt_2,296). happensAt(active(lb1gt_2),296).
orientation(lb1gt_1,151,297). coords(lb1gt_1,158,161,297).  visible(lb1gt_1,297). happensAt(walking(lb1gt_1),297). orientation(lb1gt_2,146,297). coords(lb1gt_2,194,153,297).  visible(lb1gt_2,297). happensAt(active(lb1gt_2),297).
orientation(lb1gt_1,151,298). coords(lb1gt_1,159,162,298).  visible(lb1gt_1,298). happensAt(walking(lb1gt_1),298). orientation(lb1gt_2,146,298). coords(lb1gt_2,194,153,298).  visible(lb1gt_2,298). happensAt(active(lb1gt_2),298).
orientation(lb1gt_1,151,299). coords(lb1gt_1,159,163,299).  visible(lb1gt_1,299). happensAt(walking(lb1gt_1),299). orientation(lb1gt_2,146,299). coords(lb1gt_2,194,153,299).  visible(lb1gt_2,299). happensAt(active(lb1gt_2),299).
orientation(lb1gt_1,151,300). coords(lb1gt_1,160,164,300).  visible(lb1gt_1,300). happensAt(walking(lb1gt_1),300). orientation(lb1gt_2,146,300). coords(lb1gt_2,194,152,300).  visible(lb1gt_2,300). happensAt(active(lb1gt_2),300).
orientation(lb1gt_1,151,301). coords(lb1gt_1,160,164,301).  visible(lb1gt_1,301). happensAt(walking(lb1gt_1),301). orientation(lb1gt_2,146,301). coords(lb1gt_2,194,152,301).  visible(lb1gt_2,301). happensAt(active(lb1gt_2),301).
orientation(lb1gt_1,151,302). coords(lb1gt_1,161,165,302).  visible(lb1gt_1,302). happensAt(walking(lb1gt_1),302). orientation(lb1gt_2,146,302). coords(lb1gt_2,194,152,302).  visible(lb1gt_2,302). happensAt(active(lb1gt_2),302).
orientation(lb1gt_1,151,303). coords(lb1gt_1,161,165,303).  visible(lb1gt_1,303). happensAt(walking(lb1gt_1),303). orientation(lb1gt_2,146,303). coords(lb1gt_2,194,152,303).  visible(lb1gt_2,303). happensAt(active(lb1gt_2),303).
orientation(lb1gt_1,151,304). coords(lb1gt_1,161,166,304).  visible(lb1gt_1,304). happensAt(walking(lb1gt_1),304). orientation(lb1gt_2,146,304). coords(lb1gt_2,194,152,304).  visible(lb1gt_2,304). happensAt(active(lb1gt_2),304).
orientation(lb1gt_1,151,305). coords(lb1gt_1,161,166,305).  visible(lb1gt_1,305). happensAt(walking(lb1gt_1),305). orientation(lb1gt_2,146,305). coords(lb1gt_2,194,152,305).  visible(lb1gt_2,305). happensAt(active(lb1gt_2),305).
orientation(lb1gt_1,151,306). coords(lb1gt_1,161,167,306).  visible(lb1gt_1,306). happensAt(walking(lb1gt_1),306). orientation(lb1gt_2,146,306). coords(lb1gt_2,194,152,306).  visible(lb1gt_2,306). happensAt(active(lb1gt_2),306).
orientation(lb1gt_1,151,307). coords(lb1gt_1,161,167,307).  visible(lb1gt_1,307). happensAt(walking(lb1gt_1),307). orientation(lb1gt_2,146,307). coords(lb1gt_2,194,152,307).  visible(lb1gt_2,307). happensAt(active(lb1gt_2),307).
orientation(lb1gt_1,151,308). coords(lb1gt_1,161,168,308).  visible(lb1gt_1,308). happensAt(walking(lb1gt_1),308). orientation(lb1gt_2,146,308). coords(lb1gt_2,194,152,308).  visible(lb1gt_2,308). happensAt(active(lb1gt_2),308).
orientation(lb1gt_1,151,309). coords(lb1gt_1,161,168,309).  visible(lb1gt_1,309). happensAt(walking(lb1gt_1),309). orientation(lb1gt_2,146,309). coords(lb1gt_2,194,151,309).  visible(lb1gt_2,309). happensAt(active(lb1gt_2),309).
orientation(lb1gt_1,151,310). coords(lb1gt_1,161,168,310).  visible(lb1gt_1,310). happensAt(walking(lb1gt_1),310). orientation(lb1gt_2,146,310). coords(lb1gt_2,193,151,310).  visible(lb1gt_2,310). happensAt(active(lb1gt_2),310).
orientation(lb1gt_1,151,311). coords(lb1gt_1,162,169,311).  visible(lb1gt_1,311). happensAt(walking(lb1gt_1),311). orientation(lb1gt_2,146,311). coords(lb1gt_2,193,151,311).  visible(lb1gt_2,311). happensAt(active(lb1gt_2),311).
orientation(lb1gt_1,151,312). coords(lb1gt_1,162,169,312).  visible(lb1gt_1,312). happensAt(walking(lb1gt_1),312). orientation(lb1gt_2,146,312). coords(lb1gt_2,193,151,312).  visible(lb1gt_2,312). happensAt(active(lb1gt_2),312).
#show holdsAt/2.