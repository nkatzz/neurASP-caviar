
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

	orientation(rwgt_0,42,39). coords(rwgt_0,307,278,39).  visible(rwgt_0,39). happensAt(walking(rwgt_0),39). orientation(rwgt_1,41,39). coords(rwgt_1,256,278,39).  visible(rwgt_1,39). happensAt(walking(rwgt_1),39).
orientation(rwgt_0,40,40). coords(rwgt_0,307,278,40).  visible(rwgt_0,40). happensAt(walking(rwgt_0),40). orientation(rwgt_1,35,40). coords(rwgt_1,253,275,40).  visible(rwgt_1,40). happensAt(walking(rwgt_1),40).
orientation(rwgt_0,46,41). coords(rwgt_0,307,278,41).  visible(rwgt_0,41). happensAt(walking(rwgt_0),41). orientation(rwgt_1,35,41). coords(rwgt_1,252,272,41).  visible(rwgt_1,41). happensAt(walking(rwgt_1),41).
orientation(rwgt_0,43,42). coords(rwgt_0,306,278,42).  visible(rwgt_0,42). happensAt(walking(rwgt_0),42). orientation(rwgt_1,35,42). coords(rwgt_1,251,269,42).  visible(rwgt_1,42). happensAt(walking(rwgt_1),42).
orientation(rwgt_0,38,43). coords(rwgt_0,306,278,43).  visible(rwgt_0,43). happensAt(walking(rwgt_0),43). orientation(rwgt_1,22,43). coords(rwgt_1,251,267,43).  visible(rwgt_1,43). happensAt(walking(rwgt_1),43).
orientation(rwgt_0,38,44). coords(rwgt_0,306,278,44).  visible(rwgt_0,44). happensAt(walking(rwgt_0),44). orientation(rwgt_1,22,44). coords(rwgt_1,250,266,44).  visible(rwgt_1,44). happensAt(walking(rwgt_1),44).
orientation(rwgt_0,40,45). coords(rwgt_0,305,278,45).  visible(rwgt_0,45). happensAt(walking(rwgt_0),45). orientation(rwgt_1,11,45). coords(rwgt_1,250,265,45).  visible(rwgt_1,45). happensAt(walking(rwgt_1),45).
orientation(rwgt_0,47,46). coords(rwgt_0,304,278,46).  visible(rwgt_0,46). happensAt(walking(rwgt_0),46). orientation(rwgt_1,11,46). coords(rwgt_1,248,264,46).  visible(rwgt_1,46). happensAt(walking(rwgt_1),46).
orientation(rwgt_0,44,47). coords(rwgt_0,304,278,47).  visible(rwgt_0,47). happensAt(walking(rwgt_0),47). orientation(rwgt_1,11,47). coords(rwgt_1,247,263,47).  visible(rwgt_1,47). happensAt(walking(rwgt_1),47).
orientation(rwgt_0,61,48). coords(rwgt_0,304,278,48).  visible(rwgt_0,48). happensAt(walking(rwgt_0),48). orientation(rwgt_1,3,48). coords(rwgt_1,246,262,48).  visible(rwgt_1,48). happensAt(walking(rwgt_1),48).
orientation(rwgt_0,53,49). coords(rwgt_0,304,278,49).  visible(rwgt_0,49). happensAt(walking(rwgt_0),49). orientation(rwgt_1,3,49). coords(rwgt_1,243,260,49).  visible(rwgt_1,49). happensAt(walking(rwgt_1),49).
orientation(rwgt_0,50,50). coords(rwgt_0,303,277,50).  visible(rwgt_0,50). happensAt(walking(rwgt_0),50). orientation(rwgt_1,3,50). coords(rwgt_1,242,258,50).  visible(rwgt_1,50). happensAt(walking(rwgt_1),50).
orientation(rwgt_0,49,51). coords(rwgt_0,303,276,51).  visible(rwgt_0,51). happensAt(walking(rwgt_0),51). orientation(rwgt_1,3,51). coords(rwgt_1,241,257,51).  visible(rwgt_1,51). happensAt(walking(rwgt_1),51).
orientation(rwgt_0,51,52). coords(rwgt_0,302,275,52).  visible(rwgt_0,52). happensAt(walking(rwgt_0),52). orientation(rwgt_1,3,52). coords(rwgt_1,240,256,52).  visible(rwgt_1,52). happensAt(walking(rwgt_1),52).
orientation(rwgt_0,63,53). coords(rwgt_0,301,274,53).  visible(rwgt_0,53). happensAt(walking(rwgt_0),53). orientation(rwgt_1,164,53). coords(rwgt_1,239,254,53).  visible(rwgt_1,53). happensAt(walking(rwgt_1),53).
orientation(rwgt_0,61,54). coords(rwgt_0,300,274,54).  visible(rwgt_0,54). happensAt(walking(rwgt_0),54). orientation(rwgt_1,164,54). coords(rwgt_1,238,252,54).  visible(rwgt_1,54). happensAt(walking(rwgt_1),54).
orientation(rwgt_0,40,55). coords(rwgt_0,299,274,55).  visible(rwgt_0,55). happensAt(walking(rwgt_0),55). orientation(rwgt_1,164,55). coords(rwgt_1,237,249,55).  visible(rwgt_1,55). happensAt(walking(rwgt_1),55).
orientation(rwgt_0,42,56). coords(rwgt_0,298,274,56).  visible(rwgt_0,56). happensAt(walking(rwgt_0),56). orientation(rwgt_1,164,56). coords(rwgt_1,236,245,56).  visible(rwgt_1,56). happensAt(walking(rwgt_1),56).
orientation(rwgt_0,42,57). coords(rwgt_0,296,274,57).  visible(rwgt_0,57). happensAt(walking(rwgt_0),57). orientation(rwgt_1,170,57). coords(rwgt_1,235,243,57).  visible(rwgt_1,57). happensAt(walking(rwgt_1),57).
orientation(rwgt_0,42,58). coords(rwgt_0,294,274,58).  visible(rwgt_0,58). happensAt(walking(rwgt_0),58). orientation(rwgt_1,170,58). coords(rwgt_1,235,240,58).  visible(rwgt_1,58). happensAt(walking(rwgt_1),58).
orientation(rwgt_0,42,59). coords(rwgt_0,292,274,59).  visible(rwgt_0,59). happensAt(walking(rwgt_0),59). orientation(rwgt_1,170,59). coords(rwgt_1,234,238,59).  visible(rwgt_1,59). happensAt(walking(rwgt_1),59).
orientation(rwgt_0,42,60). coords(rwgt_0,289,271,60).  visible(rwgt_0,60). happensAt(walking(rwgt_0),60). orientation(rwgt_1,170,60). coords(rwgt_1,231,236,60).  visible(rwgt_1,60). happensAt(walking(rwgt_1),60).
orientation(rwgt_0,42,61). coords(rwgt_0,287,269,61).  visible(rwgt_0,61). happensAt(walking(rwgt_0),61). orientation(rwgt_1,170,61). coords(rwgt_1,230,235,61).  visible(rwgt_1,61). happensAt(walking(rwgt_1),61).
orientation(rwgt_0,42,62). coords(rwgt_0,284,266,62).  visible(rwgt_0,62). happensAt(walking(rwgt_0),62). orientation(rwgt_1,170,62). coords(rwgt_1,228,232,62).  visible(rwgt_1,62). happensAt(walking(rwgt_1),62).
orientation(rwgt_0,42,63). coords(rwgt_0,281,261,63).  visible(rwgt_0,63). happensAt(walking(rwgt_0),63). orientation(rwgt_1,170,63). coords(rwgt_1,225,229,63).  visible(rwgt_1,63). happensAt(walking(rwgt_1),63).
#show holdsAt/2.