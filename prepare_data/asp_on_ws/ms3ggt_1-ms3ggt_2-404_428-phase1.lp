
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

	orientation(ms3ggt_1,153,404). coords(ms3ggt_1,151,159,404).  visible(ms3ggt_1,404). happensAt(walking(ms3ggt_1),404). orientation(ms3ggt_2,159,404). coords(ms3ggt_2,137,193,404).  visible(ms3ggt_2,404). happensAt(walking(ms3ggt_2),404).
orientation(ms3ggt_1,153,405). coords(ms3ggt_1,151,159,405).  visible(ms3ggt_1,405). happensAt(walking(ms3ggt_1),405). orientation(ms3ggt_2,159,405). coords(ms3ggt_2,137,191,405).  visible(ms3ggt_2,405). happensAt(walking(ms3ggt_2),405).
orientation(ms3ggt_1,153,406). coords(ms3ggt_1,152,159,406).  visible(ms3ggt_1,406). happensAt(walking(ms3ggt_1),406). orientation(ms3ggt_2,159,406). coords(ms3ggt_2,136,190,406).  visible(ms3ggt_2,406). happensAt(walking(ms3ggt_2),406).
orientation(ms3ggt_1,153,407). coords(ms3ggt_1,152,159,407).  visible(ms3ggt_1,407). happensAt(walking(ms3ggt_1),407). orientation(ms3ggt_2,159,407). coords(ms3ggt_2,136,190,407).  visible(ms3ggt_2,407). happensAt(walking(ms3ggt_2),407).
orientation(ms3ggt_1,153,408). coords(ms3ggt_1,152,159,408).  visible(ms3ggt_1,408). happensAt(walking(ms3ggt_1),408). orientation(ms3ggt_2,159,408). coords(ms3ggt_2,135,189,408).  visible(ms3ggt_2,408). happensAt(walking(ms3ggt_2),408).
orientation(ms3ggt_1,153,409). coords(ms3ggt_1,152,159,409).  visible(ms3ggt_1,409). happensAt(walking(ms3ggt_1),409). orientation(ms3ggt_2,159,409). coords(ms3ggt_2,132,187,409).  visible(ms3ggt_2,409). happensAt(walking(ms3ggt_2),409).
orientation(ms3ggt_1,153,410). coords(ms3ggt_1,153,158,410).  visible(ms3ggt_1,410). happensAt(walking(ms3ggt_1),410). orientation(ms3ggt_2,159,410). coords(ms3ggt_2,130,186,410).  visible(ms3ggt_2,410). happensAt(walking(ms3ggt_2),410).
orientation(ms3ggt_1,153,411). coords(ms3ggt_1,154,158,411).  visible(ms3ggt_1,411). happensAt(walking(ms3ggt_1),411). orientation(ms3ggt_2,159,411). coords(ms3ggt_2,129,185,411).  visible(ms3ggt_2,411). happensAt(walking(ms3ggt_2),411).
orientation(ms3ggt_1,153,412). coords(ms3ggt_1,158,159,412).  visible(ms3ggt_1,412). happensAt(walking(ms3ggt_1),412). orientation(ms3ggt_2,159,412). coords(ms3ggt_2,127,183,412).  visible(ms3ggt_2,412). happensAt(walking(ms3ggt_2),412).
orientation(ms3ggt_1,153,413). coords(ms3ggt_1,159,159,413).  visible(ms3ggt_1,413). happensAt(walking(ms3ggt_1),413). orientation(ms3ggt_2,159,413). coords(ms3ggt_2,126,182,413).  visible(ms3ggt_2,413). happensAt(walking(ms3ggt_2),413).
orientation(ms3ggt_1,160,414). coords(ms3ggt_1,161,158,414).  visible(ms3ggt_1,414). happensAt(walking(ms3ggt_1),414). orientation(ms3ggt_2,162,414). coords(ms3ggt_2,126,181,414).  visible(ms3ggt_2,414). happensAt(walking(ms3ggt_2),414).
orientation(ms3ggt_1,160,415). coords(ms3ggt_1,163,158,415).  visible(ms3ggt_1,415). happensAt(walking(ms3ggt_1),415). orientation(ms3ggt_2,162,415). coords(ms3ggt_2,126,179,415).  visible(ms3ggt_2,415). happensAt(walking(ms3ggt_2),415).
orientation(ms3ggt_1,160,416). coords(ms3ggt_1,165,158,416).  visible(ms3ggt_1,416). happensAt(walking(ms3ggt_1),416). orientation(ms3ggt_2,162,416). coords(ms3ggt_2,125,178,416).  visible(ms3ggt_2,416). happensAt(walking(ms3ggt_2),416).
orientation(ms3ggt_1,160,417). coords(ms3ggt_1,167,158,417).  visible(ms3ggt_1,417). happensAt(walking(ms3ggt_1),417). orientation(ms3ggt_2,162,417). coords(ms3ggt_2,125,177,417).  visible(ms3ggt_2,417). happensAt(walking(ms3ggt_2),417).
orientation(ms3ggt_1,160,418). coords(ms3ggt_1,169,157,418).  visible(ms3ggt_1,418). happensAt(walking(ms3ggt_1),418). orientation(ms3ggt_2,162,418). coords(ms3ggt_2,125,176,418).  visible(ms3ggt_2,418). happensAt(walking(ms3ggt_2),418).
orientation(ms3ggt_1,163,419). coords(ms3ggt_1,170,157,419).  visible(ms3ggt_1,419). happensAt(walking(ms3ggt_1),419). orientation(ms3ggt_2,166,419). coords(ms3ggt_2,125,176,419).  visible(ms3ggt_2,419). happensAt(walking(ms3ggt_2),419).
orientation(ms3ggt_1,163,420). coords(ms3ggt_1,170,158,420).  visible(ms3ggt_1,420). happensAt(walking(ms3ggt_1),420). orientation(ms3ggt_2,166,420). coords(ms3ggt_2,124,176,420).  visible(ms3ggt_2,420). happensAt(walking(ms3ggt_2),420).
orientation(ms3ggt_1,163,421). coords(ms3ggt_1,171,158,421).  visible(ms3ggt_1,421). happensAt(walking(ms3ggt_1),421). orientation(ms3ggt_2,166,421). coords(ms3ggt_2,124,176,421).  visible(ms3ggt_2,421). happensAt(walking(ms3ggt_2),421).
orientation(ms3ggt_1,163,422). coords(ms3ggt_1,172,158,422).  visible(ms3ggt_1,422). happensAt(walking(ms3ggt_1),422). orientation(ms3ggt_2,166,422). coords(ms3ggt_2,122,176,422).  visible(ms3ggt_2,422). happensAt(walking(ms3ggt_2),422).
orientation(ms3ggt_1,163,423). coords(ms3ggt_1,172,157,423).  visible(ms3ggt_1,423). happensAt(walking(ms3ggt_1),423). orientation(ms3ggt_2,166,423). coords(ms3ggt_2,121,176,423).  visible(ms3ggt_2,423). happensAt(walking(ms3ggt_2),423).
orientation(ms3ggt_1,163,424). coords(ms3ggt_1,173,157,424).  visible(ms3ggt_1,424). happensAt(walking(ms3ggt_1),424). orientation(ms3ggt_2,166,424). coords(ms3ggt_2,120,177,424).  visible(ms3ggt_2,424). happensAt(walking(ms3ggt_2),424).
orientation(ms3ggt_1,163,425). coords(ms3ggt_1,173,157,425).  visible(ms3ggt_1,425). happensAt(walking(ms3ggt_1),425). orientation(ms3ggt_2,165,425). coords(ms3ggt_2,120,177,425).  visible(ms3ggt_2,425). happensAt(walking(ms3ggt_2),425).
orientation(ms3ggt_1,163,426). coords(ms3ggt_1,174,156,426).  visible(ms3ggt_1,426). happensAt(walking(ms3ggt_1),426). orientation(ms3ggt_2,159,426). coords(ms3ggt_2,118,177,426).  visible(ms3ggt_2,426). happensAt(walking(ms3ggt_2),426).
orientation(ms3ggt_1,163,427). coords(ms3ggt_1,174,156,427).  visible(ms3ggt_1,427). happensAt(walking(ms3ggt_1),427). orientation(ms3ggt_2,159,427). coords(ms3ggt_2,117,177,427).  visible(ms3ggt_2,427). happensAt(walking(ms3ggt_2),427).
orientation(ms3ggt_1,154,428). coords(ms3ggt_1,175,157,428).  visible(ms3ggt_1,428). happensAt(walking(ms3ggt_1),428). orientation(ms3ggt_2,159,428). coords(ms3ggt_2,117,177,428).  visible(ms3ggt_2,428). happensAt(walking(ms3ggt_2),428).
#show holdsAt/2.