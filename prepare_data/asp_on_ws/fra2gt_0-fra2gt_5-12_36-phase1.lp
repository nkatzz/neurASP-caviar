
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

	orientation(fra2gt_0,159,12). coords(fra2gt_0,179,155,12).  visible(fra2gt_0,12). happensAt(walking(fra2gt_0),12). orientation(fra2gt_5,150,12). coords(fra2gt_5,51,99,12).  visible(fra2gt_5,12). happensAt(walking(fra2gt_5),12).
orientation(fra2gt_0,159,13). coords(fra2gt_0,178,154,13).  visible(fra2gt_0,13). happensAt(walking(fra2gt_0),13). orientation(fra2gt_5,150,13). coords(fra2gt_5,50,100,13).  visible(fra2gt_5,13). happensAt(walking(fra2gt_5),13).
orientation(fra2gt_0,159,14). coords(fra2gt_0,177,151,14).  visible(fra2gt_0,14). happensAt(walking(fra2gt_0),14). orientation(fra2gt_5,150,14). coords(fra2gt_5,50,100,14).  visible(fra2gt_5,14). happensAt(walking(fra2gt_5),14).
orientation(fra2gt_0,159,15). coords(fra2gt_0,177,149,15).  visible(fra2gt_0,15). happensAt(walking(fra2gt_0),15). orientation(fra2gt_5,150,15). coords(fra2gt_5,49,100,15).  visible(fra2gt_5,15). happensAt(walking(fra2gt_5),15).
orientation(fra2gt_0,159,16). coords(fra2gt_0,177,149,16).  visible(fra2gt_0,16). happensAt(walking(fra2gt_0),16). orientation(fra2gt_5,150,16). coords(fra2gt_5,48,100,16).  visible(fra2gt_5,16). happensAt(walking(fra2gt_5),16).
orientation(fra2gt_0,159,17). coords(fra2gt_0,176,147,17).  visible(fra2gt_0,17). happensAt(walking(fra2gt_0),17). orientation(fra2gt_5,150,17). coords(fra2gt_5,48,100,17).  visible(fra2gt_5,17). happensAt(walking(fra2gt_5),17).
orientation(fra2gt_0,159,18). coords(fra2gt_0,176,146,18).  visible(fra2gt_0,18). happensAt(walking(fra2gt_0),18). orientation(fra2gt_5,150,18). coords(fra2gt_5,48,100,18).  visible(fra2gt_5,18). happensAt(walking(fra2gt_5),18).
orientation(fra2gt_0,159,19). coords(fra2gt_0,176,143,19).  visible(fra2gt_0,19). happensAt(walking(fra2gt_0),19). orientation(fra2gt_5,150,19). coords(fra2gt_5,48,100,19).  visible(fra2gt_5,19). happensAt(walking(fra2gt_5),19).
orientation(fra2gt_0,159,20). coords(fra2gt_0,176,142,20).  visible(fra2gt_0,20). happensAt(walking(fra2gt_0),20). orientation(fra2gt_5,150,20). coords(fra2gt_5,48,101,20).  visible(fra2gt_5,20). happensAt(walking(fra2gt_5),20).
orientation(fra2gt_0,159,21). coords(fra2gt_0,176,141,21).  visible(fra2gt_0,21). happensAt(walking(fra2gt_0),21). orientation(fra2gt_5,150,21). coords(fra2gt_5,48,101,21).  visible(fra2gt_5,21). happensAt(walking(fra2gt_5),21).
orientation(fra2gt_0,159,22). coords(fra2gt_0,176,139,22).  visible(fra2gt_0,22). happensAt(walking(fra2gt_0),22). orientation(fra2gt_5,150,22). coords(fra2gt_5,48,101,22).  visible(fra2gt_5,22). happensAt(walking(fra2gt_5),22).
orientation(fra2gt_0,159,23). coords(fra2gt_0,176,139,23).  visible(fra2gt_0,23). happensAt(walking(fra2gt_0),23). orientation(fra2gt_5,150,23). coords(fra2gt_5,48,102,23).  visible(fra2gt_5,23). happensAt(walking(fra2gt_5),23).
orientation(fra2gt_0,159,24). coords(fra2gt_0,175,137,24).  visible(fra2gt_0,24). happensAt(walking(fra2gt_0),24). orientation(fra2gt_5,150,24). coords(fra2gt_5,48,102,24).  visible(fra2gt_5,24). happensAt(walking(fra2gt_5),24).
orientation(fra2gt_0,159,25). coords(fra2gt_0,176,136,25).  visible(fra2gt_0,25). happensAt(walking(fra2gt_0),25). orientation(fra2gt_5,150,25). coords(fra2gt_5,48,102,25).  visible(fra2gt_5,25). happensAt(walking(fra2gt_5),25).
orientation(fra2gt_0,159,26). coords(fra2gt_0,174,136,26).  visible(fra2gt_0,26). happensAt(walking(fra2gt_0),26). orientation(fra2gt_5,150,26). coords(fra2gt_5,47,103,26).  visible(fra2gt_5,26). happensAt(walking(fra2gt_5),26).
orientation(fra2gt_0,159,27). coords(fra2gt_0,173,135,27).  visible(fra2gt_0,27). happensAt(walking(fra2gt_0),27). orientation(fra2gt_5,150,27). coords(fra2gt_5,47,103,27).  visible(fra2gt_5,27). happensAt(walking(fra2gt_5),27).
orientation(fra2gt_0,159,28). coords(fra2gt_0,171,134,28).  visible(fra2gt_0,28). happensAt(walking(fra2gt_0),28). orientation(fra2gt_5,150,28). coords(fra2gt_5,46,104,28).  visible(fra2gt_5,28). happensAt(walking(fra2gt_5),28).
orientation(fra2gt_0,159,29). coords(fra2gt_0,170,134,29).  visible(fra2gt_0,29). happensAt(walking(fra2gt_0),29). orientation(fra2gt_5,150,29). coords(fra2gt_5,46,104,29).  visible(fra2gt_5,29). happensAt(walking(fra2gt_5),29).
orientation(fra2gt_0,159,30). coords(fra2gt_0,170,132,30).  visible(fra2gt_0,30). happensAt(walking(fra2gt_0),30). orientation(fra2gt_5,150,30). coords(fra2gt_5,46,104,30).  visible(fra2gt_5,30). happensAt(walking(fra2gt_5),30).
orientation(fra2gt_0,159,31). coords(fra2gt_0,169,131,31).  visible(fra2gt_0,31). happensAt(walking(fra2gt_0),31). orientation(fra2gt_5,150,31). coords(fra2gt_5,46,104,31).  visible(fra2gt_5,31). happensAt(walking(fra2gt_5),31).
orientation(fra2gt_0,151,32). coords(fra2gt_0,168,129,32).  visible(fra2gt_0,32). happensAt(walking(fra2gt_0),32). orientation(fra2gt_5,150,32). coords(fra2gt_5,45,104,32).  visible(fra2gt_5,32). happensAt(walking(fra2gt_5),32).
orientation(fra2gt_0,149,33). coords(fra2gt_0,166,129,33).  visible(fra2gt_0,33). happensAt(walking(fra2gt_0),33). orientation(fra2gt_5,150,33). coords(fra2gt_5,45,104,33).  visible(fra2gt_5,33). happensAt(walking(fra2gt_5),33).
orientation(fra2gt_0,149,34). coords(fra2gt_0,165,129,34).  visible(fra2gt_0,34). happensAt(walking(fra2gt_0),34). orientation(fra2gt_5,150,34). coords(fra2gt_5,44,104,34).  visible(fra2gt_5,34). happensAt(walking(fra2gt_5),34).
orientation(fra2gt_0,151,35). coords(fra2gt_0,164,129,35).  visible(fra2gt_0,35). happensAt(walking(fra2gt_0),35). orientation(fra2gt_5,150,35). coords(fra2gt_5,44,104,35).  visible(fra2gt_5,35). happensAt(walking(fra2gt_5),35).
orientation(fra2gt_0,154,36). coords(fra2gt_0,163,128,36).  visible(fra2gt_0,36). happensAt(walking(fra2gt_0),36). orientation(fra2gt_5,150,36). coords(fra2gt_5,44,105,36).  visible(fra2gt_5,36). happensAt(walking(fra2gt_5),36).
#show holdsAt/2.