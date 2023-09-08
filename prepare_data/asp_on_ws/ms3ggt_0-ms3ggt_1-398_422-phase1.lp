
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

	orientation(ms3ggt_0,158,398). coords(ms3ggt_0,140,138,398).  visible(ms3ggt_0,398). happensAt(walking(ms3ggt_0),398). orientation(ms3ggt_1,153,398). coords(ms3ggt_1,149,158,398).  visible(ms3ggt_1,398). happensAt(walking(ms3ggt_1),398).
orientation(ms3ggt_0,158,399). coords(ms3ggt_0,139,138,399).  visible(ms3ggt_0,399). happensAt(walking(ms3ggt_0),399). orientation(ms3ggt_1,153,399). coords(ms3ggt_1,149,158,399).  visible(ms3ggt_1,399). happensAt(walking(ms3ggt_1),399).
orientation(ms3ggt_0,158,400). coords(ms3ggt_0,139,138,400).  visible(ms3ggt_0,400). happensAt(walking(ms3ggt_0),400). orientation(ms3ggt_1,153,400). coords(ms3ggt_1,150,158,400).  visible(ms3ggt_1,400). happensAt(walking(ms3ggt_1),400).
orientation(ms3ggt_0,161,401). coords(ms3ggt_0,139,138,401).  visible(ms3ggt_0,401). happensAt(walking(ms3ggt_0),401). orientation(ms3ggt_1,153,401). coords(ms3ggt_1,150,159,401).  visible(ms3ggt_1,401). happensAt(walking(ms3ggt_1),401).
orientation(ms3ggt_0,161,402). coords(ms3ggt_0,138,138,402).  visible(ms3ggt_0,402). happensAt(walking(ms3ggt_0),402). orientation(ms3ggt_1,153,402). coords(ms3ggt_1,150,159,402).  visible(ms3ggt_1,402). happensAt(walking(ms3ggt_1),402).
orientation(ms3ggt_0,161,403). coords(ms3ggt_0,137,138,403).  visible(ms3ggt_0,403). happensAt(walking(ms3ggt_0),403). orientation(ms3ggt_1,153,403). coords(ms3ggt_1,150,159,403).  visible(ms3ggt_1,403). happensAt(walking(ms3ggt_1),403).
orientation(ms3ggt_0,156,404). coords(ms3ggt_0,136,139,404).  visible(ms3ggt_0,404). happensAt(walking(ms3ggt_0),404). orientation(ms3ggt_1,153,404). coords(ms3ggt_1,151,159,404).  visible(ms3ggt_1,404). happensAt(walking(ms3ggt_1),404).
orientation(ms3ggt_0,156,405). coords(ms3ggt_0,134,139,405).  visible(ms3ggt_0,405). happensAt(walking(ms3ggt_0),405). orientation(ms3ggt_1,153,405). coords(ms3ggt_1,151,159,405).  visible(ms3ggt_1,405). happensAt(walking(ms3ggt_1),405).
orientation(ms3ggt_0,156,406). coords(ms3ggt_0,133,139,406).  visible(ms3ggt_0,406). happensAt(walking(ms3ggt_0),406). orientation(ms3ggt_1,153,406). coords(ms3ggt_1,152,159,406).  visible(ms3ggt_1,406). happensAt(walking(ms3ggt_1),406).
orientation(ms3ggt_0,156,407). coords(ms3ggt_0,132,140,407).  visible(ms3ggt_0,407). happensAt(walking(ms3ggt_0),407). orientation(ms3ggt_1,153,407). coords(ms3ggt_1,152,159,407).  visible(ms3ggt_1,407). happensAt(walking(ms3ggt_1),407).
orientation(ms3ggt_0,158,408). coords(ms3ggt_0,131,140,408).  visible(ms3ggt_0,408). happensAt(walking(ms3ggt_0),408). orientation(ms3ggt_1,153,408). coords(ms3ggt_1,152,159,408).  visible(ms3ggt_1,408). happensAt(walking(ms3ggt_1),408).
orientation(ms3ggt_0,158,409). coords(ms3ggt_0,131,140,409).  visible(ms3ggt_0,409). happensAt(walking(ms3ggt_0),409). orientation(ms3ggt_1,153,409). coords(ms3ggt_1,152,159,409).  visible(ms3ggt_1,409). happensAt(walking(ms3ggt_1),409).
orientation(ms3ggt_0,158,410). coords(ms3ggt_0,131,140,410).  visible(ms3ggt_0,410). happensAt(walking(ms3ggt_0),410). orientation(ms3ggt_1,153,410). coords(ms3ggt_1,153,158,410).  visible(ms3ggt_1,410). happensAt(walking(ms3ggt_1),410).
orientation(ms3ggt_0,158,411). coords(ms3ggt_0,130,141,411).  visible(ms3ggt_0,411). happensAt(walking(ms3ggt_0),411). orientation(ms3ggt_1,153,411). coords(ms3ggt_1,154,158,411).  visible(ms3ggt_1,411). happensAt(walking(ms3ggt_1),411).
orientation(ms3ggt_0,158,412). coords(ms3ggt_0,130,141,412).  visible(ms3ggt_0,412). happensAt(walking(ms3ggt_0),412). orientation(ms3ggt_1,153,412). coords(ms3ggt_1,158,159,412).  visible(ms3ggt_1,412). happensAt(walking(ms3ggt_1),412).
orientation(ms3ggt_0,158,413). coords(ms3ggt_0,130,141,413).  visible(ms3ggt_0,413). happensAt(walking(ms3ggt_0),413). orientation(ms3ggt_1,153,413). coords(ms3ggt_1,159,159,413).  visible(ms3ggt_1,413). happensAt(walking(ms3ggt_1),413).
orientation(ms3ggt_0,162,414). coords(ms3ggt_0,128,142,414).  visible(ms3ggt_0,414). happensAt(walking(ms3ggt_0),414). orientation(ms3ggt_1,160,414). coords(ms3ggt_1,161,158,414).  visible(ms3ggt_1,414). happensAt(walking(ms3ggt_1),414).
orientation(ms3ggt_0,162,415). coords(ms3ggt_0,126,142,415).  visible(ms3ggt_0,415). happensAt(walking(ms3ggt_0),415). orientation(ms3ggt_1,160,415). coords(ms3ggt_1,163,158,415).  visible(ms3ggt_1,415). happensAt(walking(ms3ggt_1),415).
orientation(ms3ggt_0,162,416). coords(ms3ggt_0,125,142,416).  visible(ms3ggt_0,416). happensAt(walking(ms3ggt_0),416). orientation(ms3ggt_1,160,416). coords(ms3ggt_1,165,158,416).  visible(ms3ggt_1,416). happensAt(walking(ms3ggt_1),416).
orientation(ms3ggt_0,162,417). coords(ms3ggt_0,123,142,417).  visible(ms3ggt_0,417). happensAt(walking(ms3ggt_0),417). orientation(ms3ggt_1,160,417). coords(ms3ggt_1,167,158,417).  visible(ms3ggt_1,417). happensAt(walking(ms3ggt_1),417).
orientation(ms3ggt_0,164,418). coords(ms3ggt_0,122,143,418).  visible(ms3ggt_0,418). happensAt(walking(ms3ggt_0),418). orientation(ms3ggt_1,160,418). coords(ms3ggt_1,169,157,418).  visible(ms3ggt_1,418). happensAt(walking(ms3ggt_1),418).
orientation(ms3ggt_0,164,419). coords(ms3ggt_0,122,144,419).  visible(ms3ggt_0,419). happensAt(walking(ms3ggt_0),419). orientation(ms3ggt_1,163,419). coords(ms3ggt_1,170,157,419).  visible(ms3ggt_1,419). happensAt(walking(ms3ggt_1),419).
orientation(ms3ggt_0,164,420). coords(ms3ggt_0,120,144,420).  visible(ms3ggt_0,420). happensAt(walking(ms3ggt_0),420). orientation(ms3ggt_1,163,420). coords(ms3ggt_1,170,158,420).  visible(ms3ggt_1,420). happensAt(walking(ms3ggt_1),420).
orientation(ms3ggt_0,157,421). coords(ms3ggt_0,121,145,421).  visible(ms3ggt_0,421). happensAt(walking(ms3ggt_0),421). orientation(ms3ggt_1,163,421). coords(ms3ggt_1,171,158,421).  visible(ms3ggt_1,421). happensAt(walking(ms3ggt_1),421).
orientation(ms3ggt_0,157,422). coords(ms3ggt_0,121,145,422).  visible(ms3ggt_0,422). happensAt(walking(ms3ggt_0),422). orientation(ms3ggt_1,163,422). coords(ms3ggt_1,172,158,422).  visible(ms3ggt_1,422). happensAt(walking(ms3ggt_1),422).
#show holdsAt/2.