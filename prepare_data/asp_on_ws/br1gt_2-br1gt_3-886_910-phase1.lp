
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

	orientation(br1gt_2,2,886). coords(br1gt_2,85,204,886).  visible(br1gt_2,886). happensAt(inactive(br1gt_2),886). orientation(br1gt_3,104,886). coords(br1gt_3,288,149,886).  visible(br1gt_3,886). happensAt(walking(br1gt_3),886).
orientation(br1gt_2,2,887). coords(br1gt_2,85,204,887).  visible(br1gt_2,887). happensAt(inactive(br1gt_2),887). orientation(br1gt_3,110,887). coords(br1gt_3,287,146,887).  visible(br1gt_3,887). happensAt(walking(br1gt_3),887).
orientation(br1gt_2,2,888). coords(br1gt_2,85,203,888).  visible(br1gt_2,888). happensAt(inactive(br1gt_2),888). orientation(br1gt_3,110,888). coords(br1gt_3,287,143,888).  visible(br1gt_3,888). happensAt(walking(br1gt_3),888).
orientation(br1gt_2,2,889). coords(br1gt_2,85,203,889).  visible(br1gt_2,889). happensAt(active(br1gt_2),889). orientation(br1gt_3,110,889). coords(br1gt_3,287,139,889).  visible(br1gt_3,889). happensAt(walking(br1gt_3),889).
orientation(br1gt_2,2,890). coords(br1gt_2,85,203,890).  visible(br1gt_2,890). happensAt(active(br1gt_2),890). orientation(br1gt_3,108,890). coords(br1gt_3,288,138,890).  visible(br1gt_3,890). happensAt(walking(br1gt_3),890).
orientation(br1gt_2,2,891). coords(br1gt_2,85,202,891).  visible(br1gt_2,891). happensAt(active(br1gt_2),891). orientation(br1gt_3,106,891). coords(br1gt_3,288,137,891).  visible(br1gt_3,891). happensAt(walking(br1gt_3),891).
orientation(br1gt_2,2,892). coords(br1gt_2,85,203,892).  visible(br1gt_2,892). happensAt(active(br1gt_2),892). orientation(br1gt_3,106,892). coords(br1gt_3,288,136,892).  visible(br1gt_3,892). happensAt(walking(br1gt_3),892).
orientation(br1gt_2,2,893). coords(br1gt_2,85,203,893).  visible(br1gt_2,893). happensAt(active(br1gt_2),893). orientation(br1gt_3,101,893). coords(br1gt_3,287,136,893).  visible(br1gt_3,893). happensAt(walking(br1gt_3),893).
orientation(br1gt_2,2,894). coords(br1gt_2,85,203,894).  visible(br1gt_2,894). happensAt(active(br1gt_2),894). orientation(br1gt_3,97,894). coords(br1gt_3,287,135,894).  visible(br1gt_3,894). happensAt(walking(br1gt_3),894).
orientation(br1gt_2,2,895). coords(br1gt_2,85,202,895).  visible(br1gt_2,895). happensAt(active(br1gt_2),895). orientation(br1gt_3,97,895). coords(br1gt_3,288,134,895).  visible(br1gt_3,895). happensAt(walking(br1gt_3),895).
orientation(br1gt_2,2,896). coords(br1gt_2,85,203,896).  visible(br1gt_2,896). happensAt(active(br1gt_2),896). orientation(br1gt_3,94,896). coords(br1gt_3,288,134,896).  visible(br1gt_3,896). happensAt(walking(br1gt_3),896).
orientation(br1gt_2,2,897). coords(br1gt_2,85,202,897).  visible(br1gt_2,897). happensAt(active(br1gt_2),897). orientation(br1gt_3,94,897). coords(br1gt_3,288,133,897).  visible(br1gt_3,897). happensAt(walking(br1gt_3),897).
orientation(br1gt_2,2,898). coords(br1gt_2,85,201,898).  visible(br1gt_2,898). happensAt(active(br1gt_2),898). orientation(br1gt_3,94,898). coords(br1gt_3,288,131,898).  visible(br1gt_3,898). happensAt(walking(br1gt_3),898).
orientation(br1gt_2,2,899). coords(br1gt_2,85,200,899).  visible(br1gt_2,899). happensAt(active(br1gt_2),899). orientation(br1gt_3,94,899). coords(br1gt_3,289,129,899).  visible(br1gt_3,899). happensAt(walking(br1gt_3),899).
orientation(br1gt_2,2,900). coords(br1gt_2,85,200,900).  visible(br1gt_2,900). happensAt(active(br1gt_2),900). orientation(br1gt_3,94,900). coords(br1gt_3,289,126,900).  visible(br1gt_3,900). happensAt(walking(br1gt_3),900).
orientation(br1gt_2,2,901). coords(br1gt_2,85,200,901).  visible(br1gt_2,901). happensAt(active(br1gt_2),901). orientation(br1gt_3,94,901). coords(br1gt_3,289,124,901).  visible(br1gt_3,901). happensAt(walking(br1gt_3),901).
orientation(br1gt_2,2,902). coords(br1gt_2,85,200,902).  visible(br1gt_2,902). happensAt(active(br1gt_2),902). orientation(br1gt_3,94,902). coords(br1gt_3,290,123,902).  visible(br1gt_3,902). happensAt(walking(br1gt_3),902).
orientation(br1gt_2,2,903). coords(br1gt_2,85,200,903).  visible(br1gt_2,903). happensAt(active(br1gt_2),903). orientation(br1gt_3,94,903). coords(br1gt_3,290,119,903).  visible(br1gt_3,903). happensAt(walking(br1gt_3),903).
orientation(br1gt_2,2,904). coords(br1gt_2,85,200,904).  visible(br1gt_2,904). happensAt(active(br1gt_2),904). orientation(br1gt_3,89,904). coords(br1gt_3,290,117,904).  visible(br1gt_3,904). happensAt(walking(br1gt_3),904).
orientation(br1gt_2,2,905). coords(br1gt_2,85,200,905).  visible(br1gt_2,905). happensAt(active(br1gt_2),905). orientation(br1gt_3,89,905). coords(br1gt_3,290,115,905).  visible(br1gt_3,905). happensAt(walking(br1gt_3),905).
orientation(br1gt_2,2,906). coords(br1gt_2,85,200,906).  visible(br1gt_2,906). happensAt(active(br1gt_2),906). orientation(br1gt_3,89,906). coords(br1gt_3,291,113,906).  visible(br1gt_3,906). happensAt(walking(br1gt_3),906).
orientation(br1gt_2,2,907). coords(br1gt_2,85,200,907).  visible(br1gt_2,907). happensAt(active(br1gt_2),907). orientation(br1gt_3,89,907). coords(br1gt_3,291,113,907).  visible(br1gt_3,907). happensAt(walking(br1gt_3),907).
orientation(br1gt_2,2,908). coords(br1gt_2,85,200,908).  visible(br1gt_2,908). happensAt(active(br1gt_2),908). orientation(br1gt_3,89,908). coords(br1gt_3,291,112,908).  visible(br1gt_3,908). happensAt(walking(br1gt_3),908).
orientation(br1gt_2,2,909). coords(br1gt_2,85,200,909).  visible(br1gt_2,909). happensAt(active(br1gt_2),909). orientation(br1gt_3,89,909). coords(br1gt_3,294,112,909).  visible(br1gt_3,909). happensAt(walking(br1gt_3),909).
orientation(br1gt_2,2,910). coords(br1gt_2,85,200,910).  visible(br1gt_2,910). happensAt(active(br1gt_2),910). orientation(br1gt_3,89,910). coords(br1gt_3,292,111,910).  visible(br1gt_3,910). happensAt(walking(br1gt_3),910).
#show holdsAt/2.