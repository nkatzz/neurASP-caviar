
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

	orientation(lbbcgt_3,163,909). coords(lbbcgt_3,38,159,909).  visible(lbbcgt_3,909). happensAt(walking(lbbcgt_3),909). orientation(lbbcgt_5,9,909). coords(lbbcgt_5,224,239,909).  visible(lbbcgt_5,909). happensAt(walking(lbbcgt_5),909).
orientation(lbbcgt_3,163,910). coords(lbbcgt_3,37,159,910).  visible(lbbcgt_3,910). happensAt(walking(lbbcgt_3),910). orientation(lbbcgt_5,9,910). coords(lbbcgt_5,223,238,910).  visible(lbbcgt_5,910). happensAt(walking(lbbcgt_5),910).
orientation(lbbcgt_3,163,911). coords(lbbcgt_3,37,160,911).  visible(lbbcgt_3,911). happensAt(walking(lbbcgt_3),911). orientation(lbbcgt_5,9,911). coords(lbbcgt_5,221,236,911).  visible(lbbcgt_5,911). happensAt(walking(lbbcgt_5),911).
orientation(lbbcgt_3,163,912). coords(lbbcgt_3,37,160,912).  visible(lbbcgt_3,912). happensAt(walking(lbbcgt_3),912). orientation(lbbcgt_5,9,912). coords(lbbcgt_5,220,235,912).  visible(lbbcgt_5,912). happensAt(walking(lbbcgt_5),912).
orientation(lbbcgt_3,163,913). coords(lbbcgt_3,37,160,913).  visible(lbbcgt_3,913). happensAt(walking(lbbcgt_3),913). orientation(lbbcgt_5,9,913). coords(lbbcgt_5,218,230,913).  visible(lbbcgt_5,913). happensAt(walking(lbbcgt_5),913).
orientation(lbbcgt_3,163,914). coords(lbbcgt_3,37,161,914).  visible(lbbcgt_3,914). happensAt(walking(lbbcgt_3),914). orientation(lbbcgt_5,9,914). coords(lbbcgt_5,217,227,914).  visible(lbbcgt_5,914). happensAt(walking(lbbcgt_5),914).
orientation(lbbcgt_3,163,915). coords(lbbcgt_3,37,161,915).  visible(lbbcgt_3,915). happensAt(walking(lbbcgt_3),915). orientation(lbbcgt_5,9,915). coords(lbbcgt_5,215,225,915).  visible(lbbcgt_5,915). happensAt(walking(lbbcgt_5),915).
orientation(lbbcgt_3,163,916). coords(lbbcgt_3,37,161,916).  visible(lbbcgt_3,916). happensAt(walking(lbbcgt_3),916). orientation(lbbcgt_5,9,916). coords(lbbcgt_5,214,222,916).  visible(lbbcgt_5,916). happensAt(walking(lbbcgt_5),916).
orientation(lbbcgt_3,163,917). coords(lbbcgt_3,37,162,917).  visible(lbbcgt_3,917). happensAt(walking(lbbcgt_3),917). orientation(lbbcgt_5,9,917). coords(lbbcgt_5,212,219,917).  visible(lbbcgt_5,917). happensAt(walking(lbbcgt_5),917).
orientation(lbbcgt_3,163,918). coords(lbbcgt_3,37,164,918).  visible(lbbcgt_3,918). happensAt(walking(lbbcgt_3),918). orientation(lbbcgt_5,9,918). coords(lbbcgt_5,212,217,918).  visible(lbbcgt_5,918). happensAt(walking(lbbcgt_5),918).
orientation(lbbcgt_3,163,919). coords(lbbcgt_3,37,166,919).  visible(lbbcgt_3,919). happensAt(walking(lbbcgt_3),919). orientation(lbbcgt_5,173,919). coords(lbbcgt_5,212,214,919).  visible(lbbcgt_5,919). happensAt(walking(lbbcgt_5),919).
orientation(lbbcgt_3,163,920). coords(lbbcgt_3,37,167,920).  visible(lbbcgt_3,920). happensAt(walking(lbbcgt_3),920). orientation(lbbcgt_5,167,920). coords(lbbcgt_5,212,213,920).  visible(lbbcgt_5,920). happensAt(walking(lbbcgt_5),920).
orientation(lbbcgt_3,163,921). coords(lbbcgt_3,37,168,921).  visible(lbbcgt_3,921). happensAt(walking(lbbcgt_3),921). orientation(lbbcgt_5,167,921). coords(lbbcgt_5,212,213,921).  visible(lbbcgt_5,921). happensAt(walking(lbbcgt_5),921).
orientation(lbbcgt_3,163,922). coords(lbbcgt_3,37,169,922).  visible(lbbcgt_3,922). happensAt(walking(lbbcgt_3),922). orientation(lbbcgt_5,167,922). coords(lbbcgt_5,211,213,922).  visible(lbbcgt_5,922). happensAt(walking(lbbcgt_5),922).
orientation(lbbcgt_3,163,923). coords(lbbcgt_3,37,169,923).  visible(lbbcgt_3,923). happensAt(walking(lbbcgt_3),923). orientation(lbbcgt_5,167,923). coords(lbbcgt_5,210,212,923).  visible(lbbcgt_5,923). happensAt(walking(lbbcgt_5),923).
orientation(lbbcgt_3,163,924). coords(lbbcgt_3,37,170,924).  visible(lbbcgt_3,924). happensAt(walking(lbbcgt_3),924). orientation(lbbcgt_5,164,924). coords(lbbcgt_5,209,212,924).  visible(lbbcgt_5,924). happensAt(walking(lbbcgt_5),924).
orientation(lbbcgt_3,163,925). coords(lbbcgt_3,37,170,925).  visible(lbbcgt_3,925). happensAt(walking(lbbcgt_3),925). orientation(lbbcgt_5,164,925). coords(lbbcgt_5,208,210,925).  visible(lbbcgt_5,925). happensAt(walking(lbbcgt_5),925).
orientation(lbbcgt_3,163,926). coords(lbbcgt_3,36,170,926).  visible(lbbcgt_3,926). happensAt(walking(lbbcgt_3),926). orientation(lbbcgt_5,164,926). coords(lbbcgt_5,207,207,926).  visible(lbbcgt_5,926). happensAt(walking(lbbcgt_5),926).
orientation(lbbcgt_3,163,927). coords(lbbcgt_3,36,170,927).  visible(lbbcgt_3,927). happensAt(walking(lbbcgt_3),927). orientation(lbbcgt_5,164,927). coords(lbbcgt_5,206,204,927).  visible(lbbcgt_5,927). happensAt(walking(lbbcgt_5),927).
orientation(lbbcgt_3,163,928). coords(lbbcgt_3,36,172,928).  visible(lbbcgt_3,928). happensAt(walking(lbbcgt_3),928). orientation(lbbcgt_5,158,928). coords(lbbcgt_5,206,201,928).  visible(lbbcgt_5,928). happensAt(walking(lbbcgt_5),928).
orientation(lbbcgt_3,163,929). coords(lbbcgt_3,36,172,929).  visible(lbbcgt_3,929). happensAt(walking(lbbcgt_3),929). orientation(lbbcgt_5,158,929). coords(lbbcgt_5,205,198,929).  visible(lbbcgt_5,929). happensAt(walking(lbbcgt_5),929).
orientation(lbbcgt_3,163,930). coords(lbbcgt_3,36,173,930).  visible(lbbcgt_3,930). happensAt(walking(lbbcgt_3),930). orientation(lbbcgt_5,158,930). coords(lbbcgt_5,205,193,930).  visible(lbbcgt_5,930). happensAt(walking(lbbcgt_5),930).
orientation(lbbcgt_3,163,931). coords(lbbcgt_3,35,173,931).  visible(lbbcgt_3,931). happensAt(walking(lbbcgt_3),931). orientation(lbbcgt_5,158,931). coords(lbbcgt_5,204,190,931).  visible(lbbcgt_5,931). happensAt(walking(lbbcgt_5),931).
orientation(lbbcgt_3,163,932). coords(lbbcgt_3,35,176,932).  visible(lbbcgt_3,932). happensAt(walking(lbbcgt_3),932). orientation(lbbcgt_5,158,932). coords(lbbcgt_5,204,189,932).  visible(lbbcgt_5,932). happensAt(walking(lbbcgt_5),932).
orientation(lbbcgt_3,163,933). coords(lbbcgt_3,35,176,933).  visible(lbbcgt_3,933). happensAt(walking(lbbcgt_3),933). orientation(lbbcgt_5,158,933). coords(lbbcgt_5,203,187,933).  visible(lbbcgt_5,933). happensAt(walking(lbbcgt_5),933).
#show holdsAt/2.