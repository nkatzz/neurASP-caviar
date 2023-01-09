
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

	orientation(ms3ggt_0,156,404). coords(ms3ggt_0,136,139,404).  visible(ms3ggt_0,404). happensAt(walking(ms3ggt_0),404). orientation(ms3ggt_2,159,404). coords(ms3ggt_2,137,193,404).  visible(ms3ggt_2,404). happensAt(walking(ms3ggt_2),404).
orientation(ms3ggt_0,156,405). coords(ms3ggt_0,134,139,405).  visible(ms3ggt_0,405). happensAt(walking(ms3ggt_0),405). orientation(ms3ggt_2,159,405). coords(ms3ggt_2,137,191,405).  visible(ms3ggt_2,405). happensAt(walking(ms3ggt_2),405).
orientation(ms3ggt_0,156,406). coords(ms3ggt_0,133,139,406).  visible(ms3ggt_0,406). happensAt(walking(ms3ggt_0),406). orientation(ms3ggt_2,159,406). coords(ms3ggt_2,136,190,406).  visible(ms3ggt_2,406). happensAt(walking(ms3ggt_2),406).
orientation(ms3ggt_0,156,407). coords(ms3ggt_0,132,140,407).  visible(ms3ggt_0,407). happensAt(walking(ms3ggt_0),407). orientation(ms3ggt_2,159,407). coords(ms3ggt_2,136,190,407).  visible(ms3ggt_2,407). happensAt(walking(ms3ggt_2),407).
orientation(ms3ggt_0,158,408). coords(ms3ggt_0,131,140,408).  visible(ms3ggt_0,408). happensAt(walking(ms3ggt_0),408). orientation(ms3ggt_2,159,408). coords(ms3ggt_2,135,189,408).  visible(ms3ggt_2,408). happensAt(walking(ms3ggt_2),408).
orientation(ms3ggt_0,158,409). coords(ms3ggt_0,131,140,409).  visible(ms3ggt_0,409). happensAt(walking(ms3ggt_0),409). orientation(ms3ggt_2,159,409). coords(ms3ggt_2,132,187,409).  visible(ms3ggt_2,409). happensAt(walking(ms3ggt_2),409).
orientation(ms3ggt_0,158,410). coords(ms3ggt_0,131,140,410).  visible(ms3ggt_0,410). happensAt(walking(ms3ggt_0),410). orientation(ms3ggt_2,159,410). coords(ms3ggt_2,130,186,410).  visible(ms3ggt_2,410). happensAt(walking(ms3ggt_2),410).
orientation(ms3ggt_0,158,411). coords(ms3ggt_0,130,141,411).  visible(ms3ggt_0,411). happensAt(walking(ms3ggt_0),411). orientation(ms3ggt_2,159,411). coords(ms3ggt_2,129,185,411).  visible(ms3ggt_2,411). happensAt(walking(ms3ggt_2),411).
orientation(ms3ggt_0,158,412). coords(ms3ggt_0,130,141,412).  visible(ms3ggt_0,412). happensAt(walking(ms3ggt_0),412). orientation(ms3ggt_2,159,412). coords(ms3ggt_2,127,183,412).  visible(ms3ggt_2,412). happensAt(walking(ms3ggt_2),412).
orientation(ms3ggt_0,158,413). coords(ms3ggt_0,130,141,413).  visible(ms3ggt_0,413). happensAt(walking(ms3ggt_0),413). orientation(ms3ggt_2,159,413). coords(ms3ggt_2,126,182,413).  visible(ms3ggt_2,413). happensAt(walking(ms3ggt_2),413).
orientation(ms3ggt_0,162,414). coords(ms3ggt_0,128,142,414).  visible(ms3ggt_0,414). happensAt(walking(ms3ggt_0),414). orientation(ms3ggt_2,162,414). coords(ms3ggt_2,126,181,414).  visible(ms3ggt_2,414). happensAt(walking(ms3ggt_2),414).
orientation(ms3ggt_0,162,415). coords(ms3ggt_0,126,142,415).  visible(ms3ggt_0,415). happensAt(walking(ms3ggt_0),415). orientation(ms3ggt_2,162,415). coords(ms3ggt_2,126,179,415).  visible(ms3ggt_2,415). happensAt(walking(ms3ggt_2),415).
orientation(ms3ggt_0,162,416). coords(ms3ggt_0,125,142,416).  visible(ms3ggt_0,416). happensAt(walking(ms3ggt_0),416). orientation(ms3ggt_2,162,416). coords(ms3ggt_2,125,178,416).  visible(ms3ggt_2,416). happensAt(walking(ms3ggt_2),416).
orientation(ms3ggt_0,162,417). coords(ms3ggt_0,123,142,417).  visible(ms3ggt_0,417). happensAt(walking(ms3ggt_0),417). orientation(ms3ggt_2,162,417). coords(ms3ggt_2,125,177,417).  visible(ms3ggt_2,417). happensAt(walking(ms3ggt_2),417).
orientation(ms3ggt_0,164,418). coords(ms3ggt_0,122,143,418).  visible(ms3ggt_0,418). happensAt(walking(ms3ggt_0),418). orientation(ms3ggt_2,162,418). coords(ms3ggt_2,125,176,418).  visible(ms3ggt_2,418). happensAt(walking(ms3ggt_2),418).
orientation(ms3ggt_0,164,419). coords(ms3ggt_0,122,144,419).  visible(ms3ggt_0,419). happensAt(walking(ms3ggt_0),419). orientation(ms3ggt_2,166,419). coords(ms3ggt_2,125,176,419).  visible(ms3ggt_2,419). happensAt(walking(ms3ggt_2),419).
orientation(ms3ggt_0,164,420). coords(ms3ggt_0,120,144,420).  visible(ms3ggt_0,420). happensAt(walking(ms3ggt_0),420). orientation(ms3ggt_2,166,420). coords(ms3ggt_2,124,176,420).  visible(ms3ggt_2,420). happensAt(walking(ms3ggt_2),420).
orientation(ms3ggt_0,157,421). coords(ms3ggt_0,121,145,421).  visible(ms3ggt_0,421). happensAt(walking(ms3ggt_0),421). orientation(ms3ggt_2,166,421). coords(ms3ggt_2,124,176,421).  visible(ms3ggt_2,421). happensAt(walking(ms3ggt_2),421).
orientation(ms3ggt_0,157,422). coords(ms3ggt_0,121,145,422).  visible(ms3ggt_0,422). happensAt(walking(ms3ggt_0),422). orientation(ms3ggt_2,166,422). coords(ms3ggt_2,122,176,422).  visible(ms3ggt_2,422). happensAt(walking(ms3ggt_2),422).
orientation(ms3ggt_0,157,423). coords(ms3ggt_0,121,145,423).  visible(ms3ggt_0,423). happensAt(walking(ms3ggt_0),423). orientation(ms3ggt_2,166,423). coords(ms3ggt_2,121,176,423).  visible(ms3ggt_2,423). happensAt(walking(ms3ggt_2),423).
orientation(ms3ggt_0,157,424). coords(ms3ggt_0,120,145,424).  visible(ms3ggt_0,424). happensAt(walking(ms3ggt_0),424). orientation(ms3ggt_2,166,424). coords(ms3ggt_2,120,177,424).  visible(ms3ggt_2,424). happensAt(walking(ms3ggt_2),424).
orientation(ms3ggt_0,157,425). coords(ms3ggt_0,120,145,425).  visible(ms3ggt_0,425). happensAt(walking(ms3ggt_0),425). orientation(ms3ggt_2,165,425). coords(ms3ggt_2,120,177,425).  visible(ms3ggt_2,425). happensAt(walking(ms3ggt_2),425).
orientation(ms3ggt_0,157,426). coords(ms3ggt_0,119,145,426).  visible(ms3ggt_0,426). happensAt(walking(ms3ggt_0),426). orientation(ms3ggt_2,159,426). coords(ms3ggt_2,118,177,426).  visible(ms3ggt_2,426). happensAt(walking(ms3ggt_2),426).
orientation(ms3ggt_0,157,427). coords(ms3ggt_0,119,145,427).  visible(ms3ggt_0,427). happensAt(walking(ms3ggt_0),427). orientation(ms3ggt_2,159,427). coords(ms3ggt_2,117,177,427).  visible(ms3ggt_2,427). happensAt(walking(ms3ggt_2),427).
orientation(ms3ggt_0,157,428). coords(ms3ggt_0,118,145,428).  visible(ms3ggt_0,428). happensAt(walking(ms3ggt_0),428). orientation(ms3ggt_2,159,428). coords(ms3ggt_2,117,177,428).  visible(ms3ggt_2,428). happensAt(walking(ms3ggt_2),428).
#show holdsAt/2.